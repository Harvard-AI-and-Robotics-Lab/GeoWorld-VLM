"""
Export trained weights to a full HuggingFace model directory (safetensors).

Usage:
    python export_hf_model.py --ckpt best/trainable_state.pt --output /path/to/output
    python export_hf_model.py --ckpt epoch_3/trainable_state.pt --output /path/to/output

The script:
  1. Loads the base Gemma model
  2. Merges the trained weights (vision encoder, projector, etc.)
  3. Saves the full model + processor in HF format (safetensors)

The output directory can be used directly with eval_gemma.py:
    python eval_gemma.py --model-path /path/to/output
"""
import argparse
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

from hf_compat import patch_all_tied_weights_keys
from internvl_utils import is_internvl_model, load_internvl_tokenizer

# Must match the path used during training.
GEMMA_MODEL_PATH = os.environ.get("GEMMA_MODEL", "models/gemma-4-E4B-it")
SAVE_DIR = os.environ.get("OUTPUT_DIR", "outputs")


def main():
    parser = argparse.ArgumentParser(description="Export trained weights to full HF model")
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to trainable_state.pt (absolute or relative to save_dir)",
    )
    parser.add_argument(
        "--base-model", type=str, default=GEMMA_MODEL_PATH,
        help="Path to the base Gemma model",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for the merged HF model",
    )
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        # Try relative to save_dir first, then cwd
        candidate = os.path.join(SAVE_DIR, ckpt_path)
        if os.path.exists(candidate):
            ckpt_path = candidate

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Base model:  {args.base_model}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Output dir:  {args.output}")

    # Load base model
    print("\nLoading base model...")
    if is_internvl_model(args.base_model):
        patch_all_tied_weights_keys()
        model = AutoModel.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            use_flash_attn=False,
        )
        processor = load_internvl_tokenizer(args.base_model)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            args.base_model,
            local_files_only=True,
        )

    # Load trained weights
    print("Loading trained weights...")
    trained_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Filter out alignment MLP keys (student_align, teacher_align)
    # — these are extra modules not part of the base HF model
    model_keys = set(model.state_dict().keys())
    merged_count = 0
    skipped_keys = []

    for key, value in trained_state.items():
        # Strip "model." prefix if present (from Gemma4WanSpatialModel wrapping)
        clean_key = key
        if key.startswith("model."):
            clean_key = key[len("model."):]

        if clean_key in model_keys:
            # Get the target parameter and copy
            parts = clean_key.split(".")
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            param = getattr(module, parts[-1])
            param.data.copy_(value.to(param.dtype))
            merged_count += 1
        else:
            skipped_keys.append(key)

    print(f"\nMerged {merged_count} parameter tensors into the base model")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys (alignment MLPs, not part of base model):")
        for k in skipped_keys:
            print(f"  - {k}")

    # Save
    os.makedirs(args.output, exist_ok=True)
    print(f"\nSaving merged model to {args.output}...")
    model.save_pretrained(args.output, safe_serialization=True)
    processor.save_pretrained(args.output)

    print("Done! You can now evaluate with:")
    print(f"  python eval_gemma.py --model-path {args.output}")


if __name__ == "__main__":
    main()
