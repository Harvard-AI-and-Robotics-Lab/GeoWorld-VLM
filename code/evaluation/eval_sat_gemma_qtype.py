"""
Evaluate a Gemma-style multimodal model on SAT-v2 using SAT-style generative prompting.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset_sat_mcq import SATMCQDataset, collate_sat_mcq
from training.gemma4_dino_spatial_model import Gemma4DinoSpatialModel
from training.internvl_utils import prepare_internvl_inputs

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_split(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def keep_sat_question_type(raw_question_type: str, filter_mode: str, split_name: str) -> bool:
    if filter_mode == "all":
        return True
    target = "ego_movement" if split_name == "test" else "action_sequence"
    if filter_mode == "action_sequence":
        return raw_question_type == target
    if filter_mode == "non_action_sequence":
        return raw_question_type != target
    raise ValueError(f"Unsupported SAT question type filter: {filter_mode}")


def filter_sat_indices(split_dir: str, indices: Sequence[int], filter_mode: str, split_name: str) -> List[int]:
    if filter_mode == "all":
        return list(indices)
    ds = load_from_disk(split_dir)
    filtered = [
        int(idx)
        for idx in indices
        if keep_sat_question_type(ds[int(idx)]["question_type"], filter_mode, split_name)
    ]
    print(
        f"Filtered SAT {split_name}: {len(filtered)}/{len(indices)} samples "
        f"with sat_qtype_filter={filter_mode}"
    )
    return filtered


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"[\"'`“”‘’.,!?;:(){}\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_sat_prompt(question: str, options: Sequence[str]) -> str:
    if len(options) == 1:
        choice_format = options[0]
    elif len(options) == 2:
        choice_format = f"{options[0]}, or {options[1]}"
    else:
        choice_format = ", ".join(options[:-1]) + f", or {options[-1]}"
    return (
        f"Question: {question} "
        f"Answer the question using a single word or phrase. "
        f"Choose between the following options: {choice_format}. "
        f"Answer: "
    )


def extract_tag_answer(raw: str) -> Optional[str]:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def parse_prediction(raw_response: str, options: Sequence[str]) -> Optional[int]:
    candidates = [extract_tag_answer(raw_response), raw_response]
    normalized_options = [normalize_text(opt) for opt in options]

    for cand in candidates:
        if not cand:
            continue
        norm = normalize_text(cand)

        # If model outputs a letter despite phrase-style prompt, accept it.
        letter_match = re.search(r"\b([A-Z])\b", cand.upper())
        if letter_match:
            letter = letter_match.group(1)
            idx = LETTERS.find(letter)
            if 0 <= idx < len(options):
                return idx

        for idx, opt in enumerate(normalized_options):
            if norm == opt:
                return idx
        for idx, opt in enumerate(normalized_options):
            if opt and opt in norm:
                return idx
        for idx, opt in enumerate(normalized_options):
            if norm and norm in opt:
                return idx

    return None


@torch.no_grad()
def generate_batch(model_wrapper, batch) -> List[str]:
    prompts = [build_sat_prompt(q, opts) for q, opts in zip(batch["question_texts"], batch["options"])]

    if model_wrapper.is_internvl:
        responses = []
        for images, prompt in zip(batch["images"], prompts):
            inputs = prepare_internvl_inputs(
                model_wrapper.model,
                model_wrapper.tokenizer,
                [images],
                [prompt],
                model_wrapper.device,
            )
            generation_config = dict(max_new_tokens=64, do_sample=False)
            response = model_wrapper.model.chat(
                model_wrapper.tokenizer,
                inputs["pixel_values"],
                prompt if "<image>" in prompt else ("<image>\n" * len(images)) + prompt,
                generation_config,
                num_patches_list=[inputs["image_flags"].shape[0]],
                verbose=False,
            )
            responses.append(response.strip())
        return responses

    messages = []
    for images, prompt in zip(batch["images"], prompts):
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages.append([{"role": "user", "content": content}])

    inputs = model_wrapper.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs={
            "return_tensors": "pt",
            "padding": True,
        },
    )
    inputs = {k: v.to(model_wrapper.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    generated = model_wrapper.model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
    )
    input_len = inputs["input_ids"].shape[1]
    trimmed = generated[:, input_len:]
    return model_wrapper.processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


@torch.no_grad()
def evaluate_subset(model_wrapper, loader, subset_name: str) -> Dict:
    model_wrapper.eval()
    total = 0
    correct = 0
    per_type_total = Counter()
    per_type_correct = Counter()
    records: List[Dict] = []

    for batch in tqdm(loader, desc=f"eval {subset_name}", leave=False):
        responses = generate_batch(model_wrapper, batch)
        for i, raw_response in enumerate(responses):
            pred_idx = parse_prediction(raw_response, batch["options"][i])
            label = batch["labels"][i]
            qtype = batch["question_types"][i]
            is_correct = int(pred_idx == label)
            total += 1
            correct += is_correct
            per_type_total[qtype] += 1
            per_type_correct[qtype] += is_correct
            records.append({
                "subset": subset_name,
                "question_type": qtype,
                "correct": bool(is_correct),
                "pred_index": pred_idx,
                "label_index": label,
                "pred_answer": batch["options"][i][pred_idx] if pred_idx is not None else None,
                "label_answer": batch["options"][i][label],
                "question": batch["question_texts"][i],
                "raw_response": raw_response,
                "raw": batch["raw"][i],
            })

    metrics = {
        "subset": subset_name,
        "overall_acc": correct / max(total, 1),
        "n": total,
        "per_type": {
            qtype: {
                "acc": per_type_correct[qtype] / per_type_total[qtype],
                "n": per_type_total[qtype],
            }
            for qtype in sorted(per_type_total)
        },
        "records": records,
    }
    return metrics


def save_summary_csv(output_dir: str, summaries: List[Dict]) -> None:
    merged_total = defaultdict(int)
    merged_correct = defaultdict(float)
    val_acc = test_acc = 0.0
    val_n = test_n = 0

    for summary in summaries:
        if summary["subset"] == "val_eval":
            val_acc = summary["overall_acc"]
            val_n = summary["n"]
        elif summary["subset"] == "test":
            test_acc = summary["overall_acc"]
            test_n = summary["n"]
        for qtype, m in summary["per_type"].items():
            merged_total[qtype] += m["n"]
            merged_correct[qtype] += m["acc"] * m["n"]

    overall_n = val_n + test_n
    overall_acc = ((val_acc * val_n) + (test_acc * test_n)) / max(overall_n, 1)
    row = {
        "ActCons": round(100.0 * merged_correct["ActCons"] / max(merged_total["ActCons"], 1), 2),
        "EgoM": round(100.0 * merged_correct["EgoM"] / max(merged_total["EgoM"], 1), 2),
        "GoalAim": round(100.0 * merged_correct["GoalAim"] / max(merged_total["GoalAim"], 1), 2),
        "ObjectM": round(100.0 * merged_correct["ObjectM"] / max(merged_total["ObjectM"], 1), 2),
        "Perspect": round(100.0 * merged_correct["Perspect"] / max(merged_total["Perspect"], 1), 2),
        "val_acc": round(val_acc * 100, 2),
        "test_acc": round(test_acc * 100, 2),
        "overall_acc": round(overall_acc * 100, 2),
    }
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    fieldnames = ["ActCons", "EgoM", "GoalAim", "ObjectM", "Perspect", "val_acc", "test_acc", "overall_acc"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--sat-root", type=str, required=True)
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--sat_qtype_filter",
        type=str,
        default="all",
        choices=["all", "action_sequence", "non_action_sequence"],
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    split = load_split(args.split_file)
    sat_root = args.sat_root

    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cpu"

    model = Gemma4DinoSpatialModel(
        gemma_model_name=args.model_path,
        task_only=True,
        device=device,
    )
    model.eval()

    val_dir = os.path.join(sat_root, "val")
    test_dir = os.path.join(sat_root, "test")
    val_eval_indices = filter_sat_indices(
        val_dir,
        split["splits"]["val_eval"],
        args.sat_qtype_filter,
        "val_eval",
    )
    test_indices = filter_sat_indices(
        test_dir,
        split["splits"]["test"],
        args.sat_qtype_filter,
        "test",
    )

    subsets = [
        ("val_eval", SATMCQDataset(
            split_dir=val_dir,
            split_name="val_eval",
            indices=val_eval_indices,
        )),
        ("test", SATMCQDataset(
            split_dir=test_dir,
            split_name="test",
            indices=test_indices,
        )),
    ]

    summaries = []
    all_records = []
    for subset_name, ds in subsets:
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_sat_mcq,
            pin_memory=True,
        )
        summary = evaluate_subset(model, loader, subset_name)
        summaries.append({k: v for k, v in summary.items() if k != "records"})
        all_records.extend(summary["records"])

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    save_summary_csv(args.output_dir, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
