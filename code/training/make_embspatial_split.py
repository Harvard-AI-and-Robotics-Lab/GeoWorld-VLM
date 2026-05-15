"""
Create a stratified 2000/1640 split for EmbSpatial-Bench.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from dataset_embspatial_mcq import load_embspatial_json, RELATION_ORDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_embspatial_json(args.json_path)
    n = len(data)
    if args.train_size >= n:
        raise ValueError(f"train_size must be < total samples ({n})")

    rel_to_indices = defaultdict(list)
    for i, ex in enumerate(data):
        rel_to_indices[str(ex["relation"])].append(i)

    rng = random.Random(args.seed)
    for idxs in rel_to_indices.values():
        rng.shuffle(idxs)

    # proportional allocation with remainder fill
    alloc = {}
    remainders = []
    for rel, idxs in rel_to_indices.items():
        exact = len(idxs) * args.train_size / n
        base = int(exact)
        alloc[rel] = base
        remainders.append((exact - base, rel))
    current = sum(alloc.values())
    for _, rel in sorted(remainders, reverse=True):
        if current >= args.train_size:
            break
        alloc[rel] += 1
        current += 1

    train_indices, test_indices = [], []
    for rel, idxs in rel_to_indices.items():
        k = alloc[rel]
        train_indices.extend(idxs[:k])
        test_indices.extend(idxs[k:])

    train_indices.sort()
    test_indices.sort()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "total": n,
                    "train_size": len(train_indices),
                    "test_size": len(test_indices),
                    "seed": args.seed,
                },
                "splits": {
                    "train": train_indices,
                    "test": test_indices,
                },
                "by_relation": {
                    rel: {
                        "train": sum(1 for i in train_indices if data[i]["relation"] == rel),
                        "test": sum(1 for i in test_indices if data[i]["relation"] == rel),
                    }
                    for rel in RELATION_ORDER
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved split to {args.output}")


if __name__ == "__main__":
    main()
