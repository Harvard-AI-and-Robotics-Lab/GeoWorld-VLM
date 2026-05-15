"""
SAT-v2 MCQ dataset wrapper for AdaptVis-style training/evaluation.

This version keeps the original image cardinality per sample:
- single-image tasks return `[img]`
- two-image tasks return `[img1, img2]`
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from datasets import load_from_disk
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


TYPE_ALIASES = {
    "action_sequence": "EgoM",
    "ego_movement": "EgoM",
    "obj_movement": "ObjectM",
    "goal_aim": "GoalAim",
    "action_consequence": "ActCons",
    "action_conseq": "ActCons",
    "perspective": "Perspect",
}


def normalize_question_type(question_type: str) -> str:
    return TYPE_ALIASES.get(question_type, question_type)
@dataclass
class SatSample:
    images: List[Image.Image]
    options: List[str]
    label: int
    question_text: str
    question_type: str
    split_name: str
    raw: Dict[str, Any]


class SATMCQDataset(Dataset):
    def __init__(
        self,
        split_dir: str,
        split_name: str,
        indices: Optional[Sequence[int]] = None,
    ):
        self.ds = load_from_disk(split_dir)
        self.split_name = split_name
        self.indices = list(indices) if indices is not None else list(range(len(self.ds)))
        print(f"Loaded SAT {split_name}: {len(self.indices)} samples from {split_dir}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self.indices[idx]
        ex = self.ds[real_idx]

        images = [img.convert("RGB") for img in ex["images"]]

        answers = list(ex["answers"])
        correct_answer = ex["correct_answer"]
        label = answers.index(correct_answer)
        question_type = normalize_question_type(ex["question_type"])

        sample = SatSample(
            images=images,
            options=answers,
            label=label,
            question_text=ex["question"],
            question_type=question_type,
            split_name=self.split_name,
            raw={
                "orig_index": real_idx,
                "question_type_raw": ex["question_type"],
                "num_images": len(images),
                "correct_answer": correct_answer,
            },
        )
        return {
            "images": sample.images,
            "options": sample.options,
            "label": sample.label,
            "question_text": sample.question_text,
            "question_type": sample.question_type,
            "split_name": sample.split_name,
            "raw": sample.raw,
        }


def collate_sat_mcq(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [x["images"] for x in batch],
        "options": [x["options"] for x in batch],
        "labels": [x["label"] for x in batch],
        "question_texts": [x["question_text"] for x in batch],
        "question_types": [x["question_type"] for x in batch],
        "split_names": [x["split_name"] for x in batch],
        "raw": [x["raw"] for x in batch],
    }
