"""
Train Gemma4 + LingBot-World on the single-image spatial benchmarks.
"""
import argparse
import json
import os
import random
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from dataset_adaptvis_mcq import (
    AdaptVisUnifiedDataset, DATASET_CONFIGS, collate_mcq,
)
from gemma4_lingbot_spatial_model import Gemma4LingBotSpatialModel
from gemma4_lingbot_async_model import Gemma4LingBotAsyncModel

DATA_DIR = os.environ.get("ADAPTVIS_DATA_DIR", "data")
PROMPTS_DIR = os.environ.get("ADAPTVIS_PROMPTS_DIR", "prompts")
SAVE_DIR = os.environ.get("OUTPUT_DIR", "outputs/gemma4_lingbot_spatial")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device: str, max_samples: int = 0):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    option_token_ids = model.option_token_ids

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["images"]
        options = batch["options"]
        labels = batch["labels"]
        question_texts = batch.get("question_texts")

        inputs = model._prepare_inputs(images, options, question_texts)
        outputs = model.model(**inputs, return_dict=True, output_hidden_states=False, use_cache=False)
        logits = outputs.logits

        next_logits = logits[:, -1, :]
        letter_ids = option_token_ids.to(next_logits.device)
        option_logits = next_logits.index_select(dim=-1, index=letter_ids)
        target = torch.tensor(labels, dtype=torch.long, device=option_logits.device)
        loss_task = F.cross_entropy(option_logits, target)

        bs = len(labels)
        total += bs
        correct += (option_logits.argmax(dim=-1).cpu() == torch.tensor(labels)).sum().item()
        loss_sum += loss_task.item() * bs

        if max_samples > 0 and total >= max_samples:
            break

    return {"acc": correct / max(total, 1), "loss": loss_sum / max(total, 1), "n": total}


def build_datasets(args, wan_captions_data=None, wan_caption_embed_data=None):
    with open(args.split_file, "r") as f:
        splits = json.load(f)

    train_parts, test_parts = [], []
    for ds_name in DATASET_CONFIGS:
        if ds_name not in splits:
            print(f"  [SKIP] {ds_name} not in split file")
            continue
        ds_captions = wan_captions_data.get(ds_name, {}) if wan_captions_data else None
        ds_embeds = wan_caption_embed_data.get(ds_name, {}) if wan_caption_embed_data else None
        train_ds = AdaptVisUnifiedDataset(
            ds_name, args.data_dir, args.prompts_dir,
            indices=splits[ds_name]["train"], seed=args.seed,
            wan_captions=ds_captions,
            wan_caption_embeddings=ds_embeds,
        )
        test_ds = AdaptVisUnifiedDataset(
            ds_name, args.data_dir, args.prompts_dir,
            indices=splits[ds_name]["test"], seed=args.seed,
        )
        if len(train_ds) > 0:
            train_parts.append(train_ds)
        if len(test_ds) > 0:
            test_parts.append(test_ds)

    print(f"Train samples: {sum(len(d) for d in train_parts)}")
    print(f"Test samples: {sum(len(d) for d in test_parts)}")
    return ConcatDataset(train_parts), ConcatDataset(test_parts)


def get_trainable_params(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found.")
    return params


def plot_curves(train_log, eval_log, save_dir):
    steps = [r["step"] for r in train_log]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(steps, [r["loss"] for r in train_log], alpha=0.6, label="train")
    if eval_log:
        ax.plot([r["step"] for r in eval_log], [r["loss"] for r in eval_log], "o-", label="eval")
    ax.set_xlabel("step"); ax.set_ylabel("total loss"); ax.set_title("Total Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, [r["task"] for r in train_log], alpha=0.6)
    ax.set_xlabel("step"); ax.set_ylabel("task loss"); ax.set_title("Task Loss (CE)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, [r["align"] for r in train_log], alpha=0.6)
    ax.set_xlabel("step"); ax.set_ylabel("align loss"); ax.set_title("Align Loss (cosine)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, [r["pres"] for r in train_log], alpha=0.6)
    ax.set_xlabel("step"); ax.set_ylabel("preserve loss"); ax.set_title("Preserve Loss (MSE)")
    ax.grid(True, alpha=0.3)

    if eval_log:
        ax2 = axes[0, 0].twinx()
        ax2.plot([r["step"] for r in eval_log], [r["acc"] for r in eval_log],
                 "s--", color="green", label="eval acc")
        ax2.set_ylabel("accuracy"); ax2.legend(loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"Loss curves saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma_model_name", type=str, default=None)
    parser.add_argument("--lingbot_model_dir", type=str, default=None,
                        help="Path to lingbot-world-base-cam checkpoint dir")
    parser.add_argument("--lingbot_code_dir", type=str, default=None,
                        help="Path to lingbot-world source code dir")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--prompts_dir", type=str, default=PROMPTS_DIR)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)
    parser.add_argument("--split_file", type=str, required=True,
                        help="Path to data_split.json")

    parser.add_argument("--gpu", type=int, default=0,
                        help="Student GPU (Gemma)")
    parser.add_argument("--teacher_gpu", type=int, default=1,
                        help="Teacher GPU (LingBot-World)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_per_epoch", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--align_dim", type=int, default=512)
    parser.add_argument("--align_hidden_dim", type=int, default=1024)
    parser.add_argument("--lambda_align", type=float, default=0.2)
    parser.add_argument("--lambda_preserve", type=float, default=0.05)

    parser.add_argument("--freeze_llm", action="store_true", default=True)
    parser.add_argument("--no_freeze_llm", action="store_false", dest="freeze_llm")
    parser.add_argument("--no_unfreeze_vision", action="store_true")
    parser.add_argument("--no_unfreeze_projector", action="store_true")

    parser.add_argument("--task_only", action="store_true")
    parser.add_argument("--align_only", action="store_true")
    parser.add_argument("--wan_use_prompt", action="store_true")
    parser.add_argument("--wan_caption_file", type=str, default=None,
                        help="Path to wan_captions_v2.json")
    parser.add_argument("--wan_caption_embed_cache", type=str, default=None,
                        help="Path to precomputed T5 embeddings (.pt)")
    parser.add_argument("--prompt_embed_cache", type=str, default=None,
                        help="Path to precomputed T5 embeddings for question prompts (.pt)")

    parser.add_argument("--wan_height", type=int, default=480)
    parser.add_argument("--wan_width", type=int, default=832)
    parser.add_argument("--wan_num_frames", type=int, default=9)
    parser.add_argument("--wan_target_timestep", type=int, default=300)
    parser.add_argument("--wan_shift", type=float, default=5.0)
    parser.add_argument("--wan_hook_block_index", type=int, default=-1)
    parser.add_argument("--teacher_mode", type=str, default="static",
                        choices=["static", "single", "i2v", "jitter", "noisy_copy", "bookend"],
                        help="Teacher video construction mode")
    parser.add_argument("--jitter_strength", type=float, default=0.3,
                        help="Jitter augmentation strength for jitter mode (0-1)")
    parser.add_argument("--i2v_num_frames", type=int, default=33,
                        help="Number of frames for i2v mode (must be 4n+1)")
    parser.add_argument("--i2v_denoise_steps", type=int, default=50,
                        help="Number of denoise steps for i2v mode")
    parser.add_argument("--dual_timesteps", type=int, nargs=2, default=None,
                        help="Two timesteps for dual-timestep concat alignment, e.g. --dual_timesteps 300 700")
    parser.add_argument("--dual_blocks", type=int, nargs="+", default=None,
                        help="Multiple block indices to concat features, e.g. --dual_blocks 24 27")
    parser.add_argument("--num_teacher_steps", type=int, default=0,
                        help="Number of DiT forward passes (0=default behavior, 1=single, N>1=multi-step extract on last)")
    parser.add_argument("--use_fast_model", action="store_true",
                        help="Use lingbot-world-fast checkpoint instead of low_noise_model")
    parser.add_argument("--use_camera_perturbation", action="store_true",
                        help="Enable random WASD camera perturbation for teacher")
    parser.add_argument("--wan_prompt_text", type=str, default=None,
                        help='Fixed text prompt for teacher, e.g. "move far away"')
    parser.add_argument("--internvl_max_num", type=int, default=6,
                        help="Max dynamic image tiles for InternVL inputs")
    parser.add_argument("--async_teacher_overlap", action="store_true",
                        help="Overlap teacher execution on teacher GPU with student/preserve work on student GPU")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    if torch.cuda.is_available():
        student_device = f"cuda:{args.gpu}"
        teacher_device = f"cuda:{args.teacher_gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        student_device = "cpu"
        teacher_device = "cpu"
    device = student_device
    print(f"Student device: {student_device}, Teacher device: {teacher_device}")

    wan_captions_data = None
    if args.wan_caption_file:
        with open(args.wan_caption_file) as f:
            wan_captions_data = json.load(f)
        total_caps = sum(len(v) for v in wan_captions_data.values())
        print(f"Loaded {total_caps} captions from {args.wan_caption_file}")

    wan_caption_embed_data = None
    if args.wan_caption_embed_cache:
        wan_caption_embed_data = torch.load(args.wan_caption_embed_cache, map_location="cpu")
        total_embeds = sum(len(v) for v in wan_caption_embed_data.values())
        print(f"Loaded {total_embeds} precomputed T5 embeddings from {args.wan_caption_embed_cache}")

    if args.prompt_embed_cache:
        prompt_embed_data = torch.load(args.prompt_embed_cache, map_location="cpu")
        total_embeds = sum(len(v) for v in prompt_embed_data.values())
        print(f"Loaded {total_embeds} precomputed prompt T5 embeddings from {args.prompt_embed_cache}")
        if wan_caption_embed_data is None:
            wan_caption_embed_data = prompt_embed_data
        else:
            for ds_name, embeds in prompt_embed_data.items():
                wan_caption_embed_data.setdefault(ds_name, {}).update(embeds)

    train_ds, test_ds = build_datasets(args, wan_captions_data, wan_caption_embed_data)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_mcq, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_mcq, pin_memory=True)

    model_cls = Gemma4LingBotAsyncModel if args.async_teacher_overlap else Gemma4LingBotSpatialModel
    model = model_cls(
        gemma_model_name=args.gemma_model_name,
        lingbot_model_dir=args.lingbot_model_dir,
        lingbot_code_dir=args.lingbot_code_dir,
        align_dim=args.align_dim, align_hidden_dim=args.align_hidden_dim,
        lambda_align=args.lambda_align, lambda_preserve=args.lambda_preserve,
        freeze_llm=args.freeze_llm,
        unfreeze_vision=not args.no_unfreeze_vision,
        unfreeze_projector=not args.no_unfreeze_projector,
        wan_height=args.wan_height, wan_width=args.wan_width,
        wan_num_frames=args.wan_num_frames,
        wan_target_timestep=args.wan_target_timestep, wan_shift=args.wan_shift,
        wan_hook_block_index=args.wan_hook_block_index,
        teacher_mode=args.teacher_mode,
        i2v_num_frames=args.i2v_num_frames,
        i2v_denoise_steps=args.i2v_denoise_steps,
        dual_timesteps=args.dual_timesteps,
        dual_blocks=args.dual_blocks,
        task_only=args.task_only, align_only=args.align_only,
        wan_use_prompt=args.wan_use_prompt,
        jitter_strength=args.jitter_strength,
        num_teacher_steps=args.num_teacher_steps,
        use_fast_model=args.use_fast_model,
        use_camera_perturbation=args.use_camera_perturbation,
        wan_prompt_text=args.wan_prompt_text,
        student_device=student_device,
        teacher_device=teacher_device,
        internvl_max_num=args.internvl_max_num,
    )

    trainable_params = get_trainable_params(model)
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    history, train_log, eval_log = [], [], []
    total_steps_per_epoch = len(train_loader)
    eval_interval = max(1, total_steps_per_epoch // args.eval_per_epoch)
    eval_max_samples = max(1, len(test_ds) // 10)
    print(f"Steps/epoch: {total_steps_per_epoch}, eval every {eval_interval} steps (eval samples: {eval_max_samples})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        running = {"loss": 0.0, "task": 0.0, "align": 0.0, "pres": 0.0, "n": 0}

        for step, batch in enumerate(pbar, start=1):
            out = model(batch["images"], batch["options"], batch["labels"],
                        question_texts=batch.get("question_texts"),
                        wan_captions=batch.get("wan_captions"),
                        wan_caption_embeddings=batch.get("wan_caption_embeddings"))
            loss = out.loss / args.grad_accum_steps
            loss.backward()

            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            bs = len(batch["labels"])
            running["loss"] += out.loss.item() * bs
            running["task"] += out.loss_task.item() * bs
            running["align"] += out.loss_align.item() * bs
            running["pres"] += out.loss_preserve.item() * bs
            running["n"] += bs

            train_log.append({"step": global_step, "loss": out.loss.item(),
                              "task": out.loss_task.item(), "align": out.loss_align.item(),
                              "pres": out.loss_preserve.item()})

            pbar.set_postfix({k: f"{running[k]/running['n']:.4f}" for k in ["loss", "task", "align", "pres"]})

            if step % eval_interval == 0 and step < total_steps_per_epoch:
                metrics = evaluate(model, test_loader, device, max_samples=eval_max_samples)
                metrics["epoch"] = epoch; metrics["step"] = global_step
                eval_log.append(metrics); history.append(metrics)
                print(f"\n[mid-epoch eval] step {global_step}: acc={metrics['acc']:.4f} loss={metrics['loss']:.4f}")
                model.train()

        metrics = evaluate(model, test_loader, device, max_samples=eval_max_samples)
        metrics["epoch"] = epoch; metrics["step"] = global_step
        eval_log.append(metrics); history.append(metrics)
        print(f"\nEpoch {epoch} results:")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        ckpt_dir = os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        trainable_state = {k: v.cpu() for k, v in model.state_dict().items()
                          if k in trainable_names or k.startswith("student_align") or k.startswith("teacher_align")}
        torch.save(trainable_state, os.path.join(ckpt_dir, "trainable_state.pt"))
        try:
            saver = getattr(model, "processor", None) or getattr(model, "tokenizer", None)
            if saver is not None:
                saver.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"Warning: tokenizer/processor.save_pretrained failed: {e}")

        plot_curves(train_log, eval_log, args.save_dir)

    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.save_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.save_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"\nTraining complete!")
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
