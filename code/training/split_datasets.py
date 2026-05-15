"""
Split all 7 AdaptVis datasets into 50% train / 50% test.
Saves data_split.json containing line indices for each dataset.

Usage:
    python split_datasets.py \
        --data-dir /path/to/data \
        --prompts-dir /path/to/prompts \
        --output data_split.json
"""
import json
import os
import random
import re
import argparse
from typing import List, Dict, Any

DATASET_PROMPT_FILES = {
    "Controlled_Images_A": "Controlled_Images_A_with_answer_four_options.jsonl",
    "Controlled_Images_B": "Controlled_Images_B_with_answer_four_options.jsonl",
    "COCO_QA_one_obj": "COCO_QA_one_obj_with_answer_four_options.jsonl",
    "COCO_QA_two_obj": "COCO_QA_two_obj_with_answer_four_options.jsonl",
    "VG_QA_one_obj": "VG_QA_one_obj_with_answer_six_options.jsonl",
    "VG_QA_two_obj": "VG_QA_two_obj_with_answer_six_options.jsonl",
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--prompts-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data_split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    splits: Dict[str, Any] = {"seed": args.seed, "train_ratio": args.train_ratio}

    # Non-VSR datasets
    for ds_name, prompt_file in DATASET_PROMPT_FILES.items():
        path = os.path.join(args.prompts_dir, prompt_file)
        if not os.path.exists(path):
            print(f"  [SKIP] {ds_name}: {path} not found")
            continue
        items = load_jsonl(path)
        indices = list(range(len(items)))
        rng.shuffle(indices)
        n_train = int(len(indices) * args.train_ratio)
        splits[ds_name] = {
            "train": sorted(indices[:n_train]),
            "test": sorted(indices[n_train:]),
            "total": len(indices),
        }
        print(f"  {ds_name}: {n_train} train / {len(indices) - n_train} test (total {len(indices)})")

    # VSR
    vsr_path = os.path.join(args.data_dir, "test.jsonl")
    if os.path.exists(vsr_path):
        n_lines = sum(1 for line in open(vsr_path) if line.strip())
        indices = list(range(n_lines))
        rng.shuffle(indices)
        n_train = int(n_lines * args.train_ratio)
        splits["VSR"] = {
            "train": sorted(indices[:n_train]),
            "test": sorted(indices[n_train:]),
            "total": n_lines,
        }
        print(f"  VSR: {n_train} train / {n_lines - n_train} test (total {n_lines})")
    else:
        print(f"  [SKIP] VSR: {vsr_path} not found")

    with open(args.output, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSplit saved to {args.output}")


if __name__ == "__main__":
    main()
