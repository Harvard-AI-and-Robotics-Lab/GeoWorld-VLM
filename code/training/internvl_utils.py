import copy
from typing import Dict, List, Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import Qwen2Tokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def is_internvl_model(model_path: str) -> bool:
    return "internvl" in str(model_path).lower()


def load_internvl_tokenizer(model_path: str):
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    return tokenizer


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_pil_image(image: Image.Image, input_size=448, max_num=12) -> torch.Tensor:
    image = image.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def prepare_internvl_inputs(
    model,
    tokenizer,
    images: Sequence[Sequence[Image.Image] | Image.Image],
    prompts: Sequence[str],
    device: str,
    max_num: int = 12,
) -> Dict[str, torch.Tensor]:
    image_size = getattr(model.config, "force_image_size", None) or model.config.vision_config.image_size
    num_image_token = model.num_image_token
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    pixel_batches: List[torch.Tensor] = []
    num_patches_list: List[int] = []
    queries: List[str] = []

    for image_group, question in zip(images, prompts):
        if isinstance(image_group, Image.Image):
            sample_images = [image_group]
        else:
            sample_images = list(image_group)

        sample_patch_count = 0
        sample_patch_counts: List[int] = []
        for image in sample_images:
            pixel_values = load_pil_image(image, input_size=image_size, max_num=max_num)
            num_patches = pixel_values.shape[0]
            pixel_batches.append(pixel_values)
            sample_patch_count += num_patches
            sample_patch_counts.append(num_patches)

        if "<image>" not in question:
            question = ("<image>\n" * len(sample_images)) + question
        template = copy.deepcopy(model.conv_template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for sample_patches in sample_patch_counts:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * sample_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)
        queries.append(query)
        num_patches_list.append(sample_patch_count)

    tokenizer.padding_side = "left"
    model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
    pixel_values = torch.cat(pixel_batches, dim=0).to(device=device, dtype=torch.bfloat16)
    image_flags = torch.ones((pixel_values.shape[0], 1), dtype=torch.long, device=device)

    return {
        "pixel_values": pixel_values,
        "input_ids": model_inputs["input_ids"].to(device),
        "attention_mask": model_inputs["attention_mask"].to(device),
        "image_flags": image_flags,
    }
