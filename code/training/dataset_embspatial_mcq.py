"""
EmbSpatial-Bench dataset wrapper for Gemma/DINO/LingBot training and eval.
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


RELATION_ORDER = ["left", "right", "above", "under", "close", "far"]


@dataclass
class EmbSpatialSample:
    image: Image.Image
    options: List[str]
    label: int
    question_text: str
    relation: str
    question_id: str
    raw: Dict[str, Any]


def load_embspatial_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_base64_image(image_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_str))).convert("RGB")


class EmbSpatialMCQDataset(Dataset):
    def __init__(self, json_path: str, indices: Optional[Sequence[int]] = None):
        self.json_path = json_path
        self.samples = load_embspatial_json(json_path)
        self.indices = list(indices) if indices is not None else list(range(len(self.samples)))
        print(f"Loaded EmbSpatial: {len(self.indices)} samples from {json_path}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self.indices[idx]
        ex = self.samples[real_idx]
        image = decode_base64_image(ex["image"])
        options = list(ex["answer_options"])
        label = int(ex["answer"])
        relation = str(ex["relation"])
        sample = EmbSpatialSample(
            image=image,
            options=options,
            label=label,
            question_text=str(ex["question"]),
            relation=relation,
            question_id=str(ex["question_id"]),
            raw={
                "orig_index": real_idx,
                "data_source": ex.get("data_source"),
                "scene_id": ex.get("scene_id"),
                "objects": ex.get("objects"),
            },
        )
        return {
            "image": sample.image,
            "options": sample.options,
            "label": sample.label,
            "question_text": sample.question_text,
            "relation": sample.relation,
            "question_id": sample.question_id,
            "raw": sample.raw,
        }


def collate_embspatial_mcq(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [x["image"] for x in batch],
        "options": [x["options"] for x in batch],
        "labels": [x["label"] for x in batch],
        "question_texts": [x["question_text"] for x in batch],
        "relations": [x["relation"] for x in batch],
        "question_ids": [x["question_id"] for x in batch],
        "raw": [x["raw"] for x in batch],
    }
