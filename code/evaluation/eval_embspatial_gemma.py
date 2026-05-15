"""
Evaluate Gemma/InternVL/Qwen-style VLMs on EmbSpatial-Bench using MCQ logits.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.dataset_embspatial_mcq import EmbSpatialMCQDataset, RELATION_ORDER, collate_embspatial_mcq
from training.gemma4_dino_spatial_model import Gemma4DinoSpatialModel

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


class EmbSpatialEvaluator:
    def __init__(self, model_path: str, gpu: int = 0):
        device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.wrapper = Gemma4DinoSpatialModel(
            gemma_model_name=model_path,
            task_only=True,
            device=device,
            teacher_device=device,
        )
        self.wrapper.eval()
        self.wrapper.option_token_ids = self.wrapper.option_token_ids[:4]
        self.device = device
        self.model_name = os.path.basename(os.path.normpath(model_path))

    @torch.no_grad()
    def eval_dataset(self, dataset: EmbSpatialMCQDataset, batch_size: int = 4):
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_embspatial_mcq,
            pin_memory=True,
        )
        per_rel = {rel: {"correct": 0, "total": 0} for rel in RELATION_ORDER}
        predictions: List[Dict] = []
        total = correct = 0

        for batch in tqdm(loader, desc="eval"):
            inputs = self.wrapper._prepare_inputs(batch["images"], batch["options"], batch.get("question_texts"))
            outputs = self.wrapper.model(**inputs, return_dict=True, output_hidden_states=False, use_cache=False)
            next_logits = outputs.logits[:, -1, :]
            letter_ids = self.wrapper.option_token_ids.to(next_logits.device)
            option_logits = next_logits.index_select(dim=-1, index=letter_ids)
            preds = option_logits.argmax(dim=-1).cpu().tolist()

            for i, pred in enumerate(preds):
                label = batch["labels"][i]
                rel = batch["relations"][i]
                ok = int(pred == label)
                total += 1
                correct += ok
                per_rel[rel]["correct"] += ok
                per_rel[rel]["total"] += 1
                predictions.append(
                    {
                        "question_id": batch["question_ids"][i],
                        "relation": rel,
                        "pred": pred,
                        "label": label,
                        "correct": bool(ok),
                        "options": batch["options"][i],
                        "question": batch["question_texts"][i],
                    }
                )

        return {
            "overall_acc": correct / total if total else 0.0,
            "per_relation": {
                rel: (per_rel[rel]["correct"] / per_rel[rel]["total"] if per_rel[rel]["total"] else 0.0)
                for rel in RELATION_ORDER
            },
            "predictions": predictions,
            "correct": correct,
            "total": total,
        }


def write_summary_csv(csv_path: str, model_name: str, metrics: Dict):
    fieldnames = ["Model"] + RELATION_ORDER + ["Overall"]
    row = {"Model": model_name}
    for rel in RELATION_ORDER:
        row[rel] = f"{metrics['per_relation'][rel] * 100:.2f}"
    row["Overall"] = f"{metrics['overall_acc'] * 100:.2f}"

    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r.get("Model") != model_name]
    existing_rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.split_file, "r", encoding="utf-8") as f:
        split = json.load(f)
    test_indices = split["splits"]["test"]
    dataset = EmbSpatialMCQDataset(args.json_path, indices=test_indices)

    os.makedirs(args.output_dir, exist_ok=True)
    evaluator = EmbSpatialEvaluator(args.model_path, gpu=args.gpu)
    metrics = evaluator.eval_dataset(dataset, batch_size=args.batch_size)

    json_path = os.path.join(args.output_dir, f"{evaluator.model_name}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    write_summary_csv(csv_path, evaluator.model_name, metrics)
    print(f"Results saved to {json_path}")
    print(f"Summary CSV updated: {csv_path}")


if __name__ == "__main__":
    main()
