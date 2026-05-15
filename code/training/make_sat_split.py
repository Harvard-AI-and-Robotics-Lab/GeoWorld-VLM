"""
Create a stratified SAT split:
- val split: 3000 train + remaining eval
- test split: all test samples reserved for real-image eval
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk


def stratified_train_indices(question_types, train_size: int, seed: int):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for idx, qtype in enumerate(question_types):
        groups[qtype].append(idx)

    for idxs in groups.values():
        rng.shuffle(idxs)

    total = len(question_types)
    per_type_base = {}
    remainders = []
    assigned = 0
    for qtype, idxs in groups.items():
        exact = len(idxs) * train_size / total
        base = int(exact)
        per_type_base[qtype] = base
        remainders.append((exact - base, qtype))
        assigned += base

    remaining = train_size - assigned
    for _, qtype in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        per_type_base[qtype] += 1
        remaining -= 1

    train_indices = []
    eval_indices = []
    for qtype, idxs in groups.items():
        cutoff = per_type_base[qtype]
        train_indices.extend(idxs[:cutoff])
        eval_indices.extend(idxs[cutoff:])

    rng.shuffle(train_indices)
    rng.shuffle(eval_indices)
    return train_indices, eval_indices


def stratified_subset_indices(indices, question_types, subset_size: int, seed: int):
    if subset_size <= 0 or subset_size >= len(indices):
        return list(indices)
    rng = random.Random(seed)
    groups = defaultdict(list)
    for idx in indices:
        groups[question_types[idx]].append(idx)
    for idxs in groups.values():
        rng.shuffle(idxs)

    total = len(indices)
    per_type_base = {}
    remainders = []
    assigned = 0
    for qtype, idxs in groups.items():
        exact = len(idxs) * subset_size / total
        base = int(exact)
        per_type_base[qtype] = base
        remainders.append((exact - base, qtype))
        assigned += base

    remaining = subset_size - assigned
    for _, qtype in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        per_type_base[qtype] += 1
        remaining -= 1

    selected = []
    for qtype, idxs in groups.items():
        selected.extend(idxs[:per_type_base[qtype]])
    rng.shuffle(selected)
    return selected


def summarize(indices, question_types):
    return dict(Counter(question_types[i] for i in indices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sat-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-eval-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sat_root = Path(args.sat_root)
    val_ds = load_from_disk(str(sat_root / "val"))
    test_ds = load_from_disk(str(sat_root / "test"))

    train_indices, val_eval_indices = stratified_train_indices(
        val_ds["question_type"], train_size=args.train_size, seed=args.seed
    )
    val_eval_indices = stratified_subset_indices(
        val_eval_indices,
        val_ds["question_type"],
        subset_size=args.val_eval_size,
        seed=args.seed + 1,
    )
    test_indices = list(range(len(test_ds)))

    payload = {
        "seed": args.seed,
        "sat_root": str(sat_root),
        "train_size": len(train_indices),
        "val_eval_size": len(val_eval_indices),
        "test_size": len(test_indices),
        "splits": {
            "val_train": train_indices,
            "val_eval": val_eval_indices,
            "test": test_indices,
        },
        "summary": {
            "val_train": summarize(train_indices, val_ds["question_type"]),
            "val_eval": summarize(val_eval_indices, val_ds["question_type"]),
            "test": summarize(test_indices, test_ds["question_type"]),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved split to {out_path}")


if __name__ == "__main__":
    main()
