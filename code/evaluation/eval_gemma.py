"""
Evaluate Gemma 4 multimodal model on AdaptVis benchmark.
Supports offline inference with local model weights.
"""

import os
import re
import csv
import json
import argparse
from typing import Dict, List, Any

# Disable network access for offline mode
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor

# Compatibility shim for different Transformers versions.
try:
    from transformers import AutoModelForImageTextToText
    MODEL_CLASS = AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForMultimodalLM
    MODEL_CLASS = AutoModelForMultimodalLM
from transformers import AutoModel

from training.hf_compat import patch_all_tied_weights_keys
from training.internvl_utils import is_internvl_model, load_internvl_tokenizer, prepare_internvl_inputs


# Dataset configurations
DATASET_CONFIGS = {
    "Controlled_Images_A": {
        "options": ["left", "right", "on", "under"],
        "prompt_file": "Controlled_Images_A_with_answer_four_options.jsonl",
        "image_dir": "controlled_images",
        "type": "multiple_choice",
        "option": "four"
    },
    "Controlled_Images_B": {
        "options": ["left", "right", "front", "behind"],
        "prompt_file": "Controlled_Images_B_with_answer_four_options.jsonl",
        "image_dir": "controlled_clevr",
        "type": "multiple_choice",
        "option": "four"
    },
    "COCO_QA_one_obj": {
        "options": ["left", "right", "above", "below"],
        "prompt_file": "COCO_QA_one_obj_with_answer_four_options.jsonl",
        "image_dir": "val2017",
        "type": "multiple_choice",
        "option": "four"
    },
    "COCO_QA_two_obj": {
        "options": ["left", "right", "above", "below"],
        "prompt_file": "COCO_QA_two_obj_with_answer_four_options.jsonl",
        "image_dir": "val2017",
        "type": "multiple_choice",
        "option": "four"
    },
    "VG_QA_one_obj": {
        "options": ["left", "right", "front", "behind", "top", "bottom"],
        "prompt_file": "VG_QA_one_obj_with_answer_six_options.jsonl",
        "image_dir": "vg_images",
        "type": "multiple_choice",
        "option": "six"
    },
    "VG_QA_two_obj": {
        "options": ["left", "right", "front", "behind", "above", "below"],
        "prompt_file": "VG_QA_two_obj_with_answer_six_options.jsonl",
        "image_dir": "vg_images",
        "type": "multiple_choice",
        "option": "six"
    },
    "VSR": {
        "options": ["yes", "no"],
        "prompt_file": None,
        "image_dir": "train2017",
        "type": "binary"
    },
}


def normalize_text(text: str) -> str:
    """Normalize model response for more robust answer parsing."""
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = re.sub(r"[\"'`“”‘’.,!?;:(){}\[\]<>]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_question_text(question: str) -> str:
    """Remove dataset-specific wrappers/tags from question."""
    question = question.replace("<image>", " ").strip()
    question = question.replace("USER:", " ")
    question = question.replace("ASSISTANT:", " ")
    question = re.sub(r"\s+", " ", question)
    return question.strip()


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class GemmaEvaluator:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}...")

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Some HF multimodal checkpoints ship `processor_config.json`,
        # others only provide `preprocessor_config.json`,
        # and InternVL-style checkpoints use tokenizer + custom code instead.
        required_files = ["config.json"]
        missing = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        has_processor = os.path.exists(os.path.join(model_path, "processor_config.json"))
        has_preprocessor = os.path.exists(os.path.join(model_path, "preprocessor_config.json"))
        is_internvl = is_internvl_model(model_path)
        has_tokenizer = os.path.exists(os.path.join(model_path, "tokenizer_config.json"))
        if missing or (not is_internvl and not (has_processor or has_preprocessor)) or (is_internvl and not has_tokenizer):
            detail = []
            if missing:
                detail.append(f"missing required files: {missing}")
            if not is_internvl and not (has_processor or has_preprocessor):
                detail.append("missing processor metadata: expected processor_config.json or preprocessor_config.json")
            if is_internvl and not has_tokenizer:
                detail.append("missing tokenizer metadata: expected tokenizer_config.json")
            raise FileNotFoundError(
                f"Model directory is incomplete. {'; '.join(detail)}"
            )

        self.model_path = model_path
        self.model_name = os.path.basename(os.path.normpath(model_path))
        self.is_internvl = is_internvl_model(model_path)

        if self.is_internvl:
            patch_all_tied_weights_keys()
            self.tokenizer = load_internvl_tokenizer(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True
            )

        # Choose dtype
        if torch.cuda.is_available():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float32

        if self.is_internvl:
            patch_all_tied_weights_keys()
            self.model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=model_dtype,
                device_map="auto",
                trust_remote_code=True,
                use_flash_attn=False,
            )
        else:
            self.model = MODEL_CLASS.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=model_dtype,
                device_map="auto"
            )
        self.model.eval()

        # Use the device of the first parameter as input device
        self.input_device = next(self.model.parameters()).device
        print(f"Model loaded successfully! input_device={self.input_device}")

    def generate_response(self, image: Image.Image, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate response for a single image-question pair."""
        if self.is_internvl:
            inputs = prepare_internvl_inputs(
                self.model,
                self.tokenizer,
                [image],
                [prompt],
                str(self.input_device),
            )
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
            response = self.model.chat(
                self.tokenizer,
                inputs["pixel_values"],
                prompt if "<image>" in prompt else "<image>\n" + prompt,
                generation_config,
                num_patches_list=[inputs["image_flags"].shape[0]],
                verbose=False,
            )
            return response.strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.input_device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        generated_tokens = outputs[0][input_len:]

        # Decode behavior can differ across processor versions.
        if hasattr(self.processor, "decode"):
            response = self.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def parse_prediction(self, response: str, options: List[str]) -> str:
        """Parse model response to extract prediction more robustly."""
        text = normalize_text(response)

        # Prefer the first generated word.
        tokens = text.split()
        if tokens:
            first = tokens[0]
            for opt in options:
                if first == opt.lower():
                    return opt.lower()

        # Then use word-boundary matching to avoid false matches such as bright/right.
        for opt in options:
            pattern = rf"\b{re.escape(opt.lower())}\b"
            if re.search(pattern, text):
                return opt.lower()

        # Extra fallback for yes/no tasks.
        if "yes" in options and text.startswith("yes"):
            return "yes"
        if "no" in options and text.startswith("no"):
            return "no"

        return ""


def load_image_mapping(data_dir: str, dataset_name: str) -> Dict[int, str]:
    """Load mapping from item_id to image path for controlled datasets."""
    if dataset_name.startswith("Controlled_Images"):
        if dataset_name == "Controlled_Images_A":
            json_file = os.path.join(data_dir, "controlled_images_dataset.json")
        else:
            json_file = os.path.join(data_dir, "controlled_clevr_dataset.json")

        if os.path.exists(json_file):
            data = load_json(json_file)
            mapping = {}
            for idx, item in enumerate(data):
                img_path = item.get("image_path", "")
                if img_path.startswith("data/"):
                    img_path = os.path.join(data_dir, img_path[5:])
                mapping[idx] = img_path
            return mapping

    return {}


def load_vsr_data(data_dir: str, split: str = "test") -> List[Dict[str, Any]]:
    """Load VSR dataset."""
    vsr_file = os.path.join(data_dir, f"{split}.jsonl")

    if not os.path.exists(vsr_file):
        print(f"VSR file not found: {vsr_file}")
        return []

    items = []
    with open(vsr_file, "r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            image_link = data.get("image_link", "")
            match = re.search(r"\.org/(.*)", image_link)
            if not match:
                continue

            image_path = os.path.join(data_dir, match.group(1))
            items.append({
                "id": idx,
                "image_path": image_path,
                "caption": data.get("caption", ""),
                "label": data.get("label", 0)
            })

    return items


def preload_dataset_aux(data_dir: str, dataset_name: str):
    """Preload auxiliary JSON files once per dataset instead of once per sample."""
    if dataset_name == "COCO_QA_one_obj":
        path = os.path.join(data_dir, "coco_qa_one_obj.json")
        return load_json(path) if os.path.exists(path) else None

    if dataset_name == "COCO_QA_two_obj":
        path = os.path.join(data_dir, "coco_qa_two_obj.json")
        return load_json(path) if os.path.exists(path) else None

    if dataset_name == "VG_QA_one_obj":
        path = os.path.join(data_dir, "vg_qa_one_obj.json")
        return load_json(path) if os.path.exists(path) else None

    if dataset_name == "VG_QA_two_obj":
        path = os.path.join(data_dir, "vg_qa_two_obj.json")
        return load_json(path) if os.path.exists(path) else None

    return None


def resolve_image_path(
    dataset_name: str,
    item_id: int,
    data_dir: str,
    image_mapping: Dict[int, str],
    aux_data
) -> str:
    """Resolve image path for each dataset type."""
    if dataset_name.startswith("Controlled_Images"):
        return image_mapping.get(item_id, "")

    if dataset_name.startswith("COCO_QA"):
        if aux_data is None or item_id >= len(aux_data):
            return ""
        img_id = str(aux_data[item_id][0]).zfill(12)
        return os.path.join(data_dir, "val2017", f"{img_id}.jpg")

    if dataset_name.startswith("VG_QA"):
        if aux_data is None or item_id >= len(aux_data):
            return ""
        img_id = aux_data[item_id][0]
        return os.path.join(data_dir, "vg_images", f"{img_id}.jpg")

    return ""


def build_prompt(item: Dict[str, Any]) -> str:
    """Build evaluation prompt."""
    if item.get("caption"):
        return (
            f'Look at this image and determine if the following statement is correct: '
            f'"{item["caption"]}" '
            f"Answer with only 'yes' or 'no'."
        )

    question = clean_question_text(item["question"])
    options_str = ", ".join(item["options"])
    return f"{question} Answer with only one word from these options: {options_str}."


def evaluate_dataset(evaluator: GemmaEvaluator, dataset_name: str, data_dir: str,
                     prompts_dir: str, test_indices=None):
    """Evaluate model on a single dataset."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {dataset_name}")
    print(f"{'=' * 60}")

    config = DATASET_CONFIGS[dataset_name]

    # Load prompts
    prompts = []
    if config.get("prompt_file"):
        prompt_path = os.path.join(prompts_dir, config["prompt_file"])
        if not os.path.exists(prompt_path):
            print(f"Prompt file not found: {prompt_path}")
            return {}
        prompts = load_jsonl(prompt_path)
        print(f"Loaded {len(prompts)} prompts")

    # Load image mapping and auxiliary metadata
    image_mapping = load_image_mapping(data_dir, dataset_name)
    aux_data = preload_dataset_aux(data_dir, dataset_name)

    items = []

    if dataset_name == "VSR":
        vsr_data = load_vsr_data(data_dir)
        print(f"Loaded {len(vsr_data)} VSR items")

        for vsr_item in vsr_data:
            image_path = vsr_item["image_path"]
            if not os.path.exists(image_path):
                continue

            gt = "yes" if vsr_item["label"] == 1 else "no"
            items.append({
                "id": vsr_item["id"],
                "image_path": image_path,
                "question": "",
                "ground_truth": gt,
                "options": ["yes", "no"],
                "caption": vsr_item["caption"]
            })
    else:
        for prompt_data in prompts:
            item_id = prompt_data["id"]
            question = prompt_data["question"]
            answer = prompt_data["answer"]

            image_path = resolve_image_path(
                dataset_name=dataset_name,
                item_id=item_id,
                data_dir=data_dir,
                image_mapping=image_mapping,
                aux_data=aux_data
            )

            if not image_path or not os.path.exists(image_path):
                continue

            if isinstance(answer, list):
                ground_truth = str(answer[0]).lower()
            else:
                ground_truth = str(answer).lower()

            items.append({
                "id": item_id,
                "image_path": image_path,
                "question": question,
                "ground_truth": ground_truth,
                "options": config["options"],
                "caption": ""
            })

    print(f"Found {len(items)} valid items to evaluate")

    # Filter by test split if provided
    if test_indices is not None:
        idx_set = set(test_indices)
        items = [it for i, it in enumerate(items) if i in idx_set]
        print(f"After split filter: {len(items)} items (test split)")

    if not items:
        print("No valid items to evaluate!")
        return {}

    results = []
    correct = 0

    for item in tqdm(items, desc=f"Evaluating {dataset_name}"):
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            continue

        prompt = build_prompt(item)

        try:
            response = evaluator.generate_response(image, prompt)
            prediction = evaluator.parse_prediction(response, item["options"])
        except Exception as e:
            print(f"Generation failed on item {item['id']}: {e}")
            response = f"[ERROR] {str(e)}"
            prediction = ""

        is_correct = prediction == item["ground_truth"]
        if is_correct:
            correct += 1

        results.append({
            "id": item["id"],
            "prediction": prediction,
            "ground_truth": item["ground_truth"],
            "response": response,
            "is_correct": is_correct,
            "image_path": item["image_path"]
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

    option_stats = {}
    for opt in config["options"]:
        opt_lower = opt.lower()
        opt_results = [r for r in results if r["ground_truth"] == opt_lower]
        if opt_results:
            opt_correct = sum(1 for r in opt_results if r["is_correct"])
            option_stats[opt_lower] = {
                "correct": opt_correct,
                "total": len(opt_results),
                "accuracy": opt_correct / len(opt_results)
            }

    print(f"\nPer-option accuracy:")
    for opt, stats in option_stats.items():
        print(f"  {opt}: {stats['accuracy'] * 100:.2f}% ({stats['correct']}/{stats['total']})")

    return {
        "dataset": dataset_name,
        "model": evaluator.model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "option_stats": option_stats,
        "results": results
    }


def save_summary_csv(csv_file: str, model_name: str, all_results: Dict[str, Any]):
    """Append summary results to CSV, deduplicating by model name (keep latest)."""
    ordered_datasets = [
        "Controlled_Images_A",
        "Controlled_Images_B",
        "COCO_QA_one_obj",
        "COCO_QA_two_obj",
        "VG_QA_one_obj",
        "VG_QA_two_obj",
        "VSR",
    ]
    fieldnames = ["Model"] + ordered_datasets + ["Overall"]

    total_correct = 0
    total_samples = 0
    row = {"Model": model_name}

    for ds in ordered_datasets:
        if ds in all_results and all_results[ds]:
            acc = all_results[ds]["accuracy"] * 100
            row[ds] = f"{acc:.2f}"
            total_correct += all_results[ds]["correct"]
            total_samples += all_results[ds]["total"]
        else:
            row[ds] = "N/A"

    overall = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    row["Overall"] = f"{overall:.2f}"

    # Read existing rows, deduplicate by model name
    existing_rows = []
    if os.path.exists(csv_file):
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r.get("Model") != model_name]

    existing_rows.append(row)

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemma on AdaptVis benchmark")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("GEMMA_MODEL", "models/gemma-4-E4B-it"),
        help="Path to local Gemma model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--prompts-dir",
        type=str,
        default="prompts",
        help="Path to prompts directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all"] + list(DATASET_CONFIGS.keys()),
        help="Dataset to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to data_split.json; when set, only evaluate the test split"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(script_dir, model_path)

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(script_dir, args.data_dir)
    prompts_dir = args.prompts_dir if os.path.isabs(args.prompts_dir) else os.path.join(script_dir, args.prompts_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(script_dir, args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Load split file if provided
    split_data = None
    if args.split_file:
        with open(args.split_file, "r") as f:
            split_data = json.load(f)
        print(f"Using test split from {args.split_file}")

    evaluator = GemmaEvaluator(model_path)

    if args.dataset == "all":
        datasets_to_eval = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_eval = [args.dataset]

    all_results = {}
    for dataset_name in datasets_to_eval:
        test_indices = None
        if split_data and dataset_name in split_data:
            test_indices = split_data[dataset_name].get("test")
        result = evaluate_dataset(
            evaluator=evaluator,
            dataset_name=dataset_name,
            data_dir=data_dir,
            prompts_dir=prompts_dir,
            test_indices=test_indices,
        )
        all_results[dataset_name] = result

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset_name, result in all_results.items():
        if result:
            print(f"{dataset_name}: {result['accuracy'] * 100:.2f}% ({result['correct']}/{result['total']})")

    output_json = os.path.join(output_dir, f"{evaluator.model_name}_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {output_json}")

    csv_file = os.path.join(output_dir, "evaluation_results.csv")
    save_summary_csv(csv_file, evaluator.model_name, all_results)
    print(f"Results appended to CSV: {csv_file}")


if __name__ == "__main__":
    main()
