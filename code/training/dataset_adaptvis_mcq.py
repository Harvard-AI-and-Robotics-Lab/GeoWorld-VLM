"""
AdaptVis MCQ Dataset for training.
Supports both the legacy [[image_id, correct, wrong]] format
and the unified JSONL prompt format used by eval (all 7 datasets).
"""
import os
import json
import re
import random
from typing import Any, Dict, List, Optional

from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Allow loading truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Dataset configs (same options as eval_gemma.py)
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "Controlled_Images_A": {
        "options": ["left", "right", "on", "under"],
        "prompt_file": "Controlled_Images_A_with_answer_four_options.jsonl",
    },
    "Controlled_Images_B": {
        "options": ["left", "right", "front", "behind"],
        "prompt_file": "Controlled_Images_B_with_answer_four_options.jsonl",
    },
    "COCO_QA_one_obj": {
        "options": ["left", "right", "above", "below"],
        "prompt_file": "COCO_QA_one_obj_with_answer_four_options.jsonl",
    },
    "COCO_QA_two_obj": {
        "options": ["left", "right", "above", "below"],
        "prompt_file": "COCO_QA_two_obj_with_answer_four_options.jsonl",
    },
    "VG_QA_one_obj": {
        "options": ["left", "right", "front", "behind", "top", "bottom"],
        "prompt_file": "VG_QA_one_obj_with_answer_six_options.jsonl",
    },
    "VG_QA_two_obj": {
        "options": ["left", "right", "front", "behind", "above", "below"],
        "prompt_file": "VG_QA_two_obj_with_answer_six_options.jsonl",
    },
    "VSR": {
        "options": ["yes", "no"],
        "prompt_file": None,
    },
}


class AdaptVisMCQDataset(Dataset):
    """
    AdaptVis VG_QA_one_obj / VG_QA_two_obj style dataset.
    Format: [[image_id, correct_caption, wrong_caption], ...]

    Supports two-option and multi-option variants.
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        num_options: int = 2,
        correct_index: int = 0,
        shuffle_options: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.json_path = json_path
        self.image_dir = image_dir
        self.num_options = num_options
        self.correct_index = correct_index
        self.shuffle_options = shuffle_options
        self.seed = seed
        self.rng = random.Random(seed)

        # Load metadata.
        self.samples = self._load_data(json_path)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {json_path}")

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        samples = []
        for item in raw_data:
            if len(item) >= 3:
                image_id = str(item[0])
                correct_caption = item[1]
                wrong_caption = item[2]

                samples.append({
                    'image_id': image_id,
                    'correct_caption': correct_caption,
                    'wrong_caption': wrong_caption,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_id = sample['image_id']

        # Load image.
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, f"{image_id}.png")

        image = Image.open(image_path).convert('RGB')

        # Build answer options.
        options = [sample['correct_caption'], sample['wrong_caption']]
        label = 0

        # Optionally shuffle answer choices.
        if self.shuffle_options:
            paired = list(zip(options, [0, 1]))
            self.rng.shuffle(paired)
            options, indices = zip(*paired)
            options = list(options)
            label = indices.index(0)

        return {
            'image': image,
            'options': options,
            'label': label,
            'image_id': image_id,
            'raw': sample,
        }


class ConcatAdaptVisDataset(Dataset):
    """Concatenate multiple AdaptVis datasets."""

    def __init__(self, datasets: List[AdaptVisMCQDataset]):
        self.datasets = datasets
        self.cumulative_sizes = [0]
        for ds in datasets:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(ds))

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Find the corresponding dataset.
        for i, ds in enumerate(self.datasets):
            if idx < self.cumulative_sizes[i + 1]:
                local_idx = idx - self.cumulative_sizes[i]
                return ds[local_idx]
        raise IndexError(f"Index {idx} out of range")


def collate_mcq(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for MCQ datasets"""
    return {
        'images': [x['image'] for x in batch],
        'options': [x['options'] for x in batch],
        'labels': [x['label'] for x in batch],
        'image_ids': [x.get('image_id', '') for x in batch],
        'question_texts': [x.get('question_text', '') for x in batch],
        'wan_captions': [x.get('wan_caption', '') for x in batch],
        'wan_caption_embeddings': [x.get('wan_caption_embedding', None) for x in batch],
        'raw': [x.get('raw', {}) for x in batch],
    }


# ---------------------------------------------------------------------------
# Helpers reused from eval_gemma.py (image path resolution)
# ---------------------------------------------------------------------------

def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _load_aux(dataset_name: str, data_dir: str):
    """Load auxiliary JSON for image-id resolution."""
    name_to_file = {
        "COCO_QA_one_obj": "coco_qa_one_obj.json",
        "COCO_QA_two_obj": "coco_qa_two_obj.json",
        "VG_QA_one_obj": "vg_qa_one_obj.json",
        "VG_QA_two_obj": "vg_qa_two_obj.json",
    }
    fname = name_to_file.get(dataset_name)
    if fname:
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            return _load_json(p)
    return None


def _load_image_mapping(dataset_name: str, data_dir: str) -> Dict[int, str]:
    if dataset_name == "Controlled_Images_A":
        jf = os.path.join(data_dir, "controlled_images_dataset.json")
    elif dataset_name == "Controlled_Images_B":
        jf = os.path.join(data_dir, "controlled_clevr_dataset.json")
    else:
        return {}
    if not os.path.exists(jf):
        return {}
    data = _load_json(jf)
    mapping = {}
    for idx, item in enumerate(data):
        ip = item.get("image_path", "")
        if ip.startswith("data/"):
            ip = os.path.join(data_dir, ip[5:])
        mapping[idx] = ip
    return mapping


def _resolve_image_path(dataset_name: str, item_id: int,
                        data_dir: str, image_mapping, aux_data) -> str:
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


def _clean_question(q: str) -> str:
    q = q.replace("<image>", " ").strip()
    q = q.replace("USER:", " ").replace("ASSISTANT:", " ")
    return re.sub(r"\s+", " ", q).strip()


# ---------------------------------------------------------------------------
# Unified dataset: reads from JSONL prompts for any of the 7 datasets
# ---------------------------------------------------------------------------

class AdaptVisUnifiedDataset(Dataset):
    """
    Reads JSONL prompt files (same format as eval) and returns MCQ items.
    Supports split by providing a list of line indices to keep.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        prompts_dir: str,
        indices: Optional[List[int]] = None,
        seed: int = 42,
        wan_captions: Optional[Dict[str, str]] = None,
        wan_caption_embeddings: Optional[Dict[str, "torch.Tensor"]] = None,
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        cfg = DATASET_CONFIGS[dataset_name]
        self.options = cfg["options"]
        self.rng = random.Random(seed)
        self.wan_captions = wan_captions or {}  # {str(index): caption}
        self.wan_caption_embeddings = wan_caption_embeddings or {}  # {str(index): tensor}

        if dataset_name == "VSR":
            all_items = self._load_vsr(data_dir)
        else:
            all_items = self._load_prompt_dataset(
                dataset_name, data_dir, prompts_dir, cfg["prompt_file"]
            )

        # filter by split indices, keeping track of original index
        if indices is not None:
            idx_set = set(indices)
            self.items = [(i, all_items[i]) for i in range(len(all_items))
                          if i in idx_set and all_items[i] is not None]
        else:
            self.items = [(i, it) for i, it in enumerate(all_items) if it is not None]

        print(f"  {dataset_name}: {len(self.items)} samples loaded")

    def _load_prompt_dataset(self, name, data_dir, prompts_dir, prompt_file):
        prompt_path = os.path.join(prompts_dir, prompt_file)
        if not os.path.exists(prompt_path):
            print(f"  Warning: {prompt_path} not found")
            return []
        prompts = _load_jsonl(prompt_path)
        aux = _load_aux(name, data_dir)
        img_map = _load_image_mapping(name, data_dir)

        items: List[Optional[Dict]] = []
        for pd in prompts:
            item_id = pd["id"]
            img_path = _resolve_image_path(name, item_id, data_dir, img_map, aux)
            if not img_path or not os.path.exists(img_path):
                items.append(None)
                continue
            answer = pd["answer"]
            if isinstance(answer, list):
                answer = str(answer[0]).lower()
            else:
                answer = str(answer).lower()
            question = _clean_question(pd.get("question", ""))
            items.append({
                "image_path": img_path,
                "question": question,
                "answer": answer,
            })
        return items

    def _load_vsr(self, data_dir):
        vsr_path = os.path.join(data_dir, "test.jsonl")
        if not os.path.exists(vsr_path):
            print(f"  Warning: {vsr_path} not found")
            return []
        items: List[Optional[Dict]] = []
        with open(vsr_path) as f:
            for line in f:
                data = json.loads(line.strip())
                link = data.get("image_link", "")
                m = re.search(r"\.org/(.*)", link)
                if not m:
                    items.append(None)
                    continue
                img_path = os.path.join(data_dir, m.group(1))
                if not os.path.exists(img_path):
                    items.append(None)
                    continue
                items.append({
                    "image_path": img_path,
                    "question": f'Is the following statement correct: "{data.get("caption", "")}"',
                    "answer": "yes" if data.get("label", 0) == 1 else "no",
                })
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        orig_idx, it = self.items[idx]
        image = Image.open(it["image_path"]).convert("RGB")

        # build options and find label
        options = list(self.options)
        answer = it["answer"]
        if answer not in [o.lower() for o in options]:
            options = [answer] + [o for o in options if o.lower() != answer]
        label = next(i for i, o in enumerate(options) if o.lower() == answer)

        # wan caption (pre-generated by Gemini) if available
        wan_caption = self.wan_captions.get(str(orig_idx), "")
        wan_caption_embedding = self.wan_caption_embeddings.get(str(orig_idx), None)

        return {
            "image": image,
            "options": options,
            "label": label,
            "question_text": it["question"],
            "wan_caption": wan_caption,
            "wan_caption_embedding": wan_caption_embedding,
            "image_id": os.path.basename(it["image_path"]),
        }
