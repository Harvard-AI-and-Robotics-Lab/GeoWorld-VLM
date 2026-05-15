"""
Train Gemma4 + LingBot-World on SAT-v2 with explicit double-image alignment modes.

Modes for two-image samples:
- pairwise_single_teacher:
    teacher runs twice in single-image mode (img1, img2) and student keeps two
    image tokens; losses are averaged by the cosine loss reduction.
- mean_pool:
    student keeps two image tokens, applies student MLP token-wise, then mean-pools
    to one vector and aligns to a single teacher feature computed from the two-image
    input in i2v_pair mode.

Single-image samples always use the standard single-image LingBot path.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import contextmanager
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk

from dataset_sat_mcq import SATMCQDataset, collate_sat_mcq
from gemma4_lingbot_spatial_model import (
    ForwardOutput,
    Gemma4LingBotSpatialModel,
    _cosine_align,
    _normalize_mse,
)

SAT_ROOT = os.environ.get("SAT_ROOT", "data/sat")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def keep_sat_question_type(raw_question_type: str, filter_mode: str, split_name: str) -> bool:
    if filter_mode == "all":
        return True
    is_test = split_name == "test"
    target = "ego_movement" if is_test else "action_sequence"
    if filter_mode == "action_sequence":
        return raw_question_type == target
    if filter_mode == "non_action_sequence":
        return raw_question_type != target
    raise ValueError(f"Unsupported SAT question type filter: {filter_mode}")


def filter_sat_indices(split_dir: str, indices: Sequence[int], filter_mode: str, split_name: str) -> List[int]:
    if filter_mode == "all":
        return list(indices)
    ds = load_from_disk(split_dir)
    filtered = [
        int(idx)
        for idx in indices
        if keep_sat_question_type(ds[int(idx)]["question_type"], filter_mode, split_name)
    ]
    print(
        f"Filtered SAT {split_name}: {len(filtered)}/{len(indices)} samples "
        f"with sat_qtype_filter={filter_mode}"
    )
    return filtered


def build_datasets(args):
    split = load_split(args.split_file)
    val_dir = os.path.join(args.sat_root, "val")
    train_indices = filter_sat_indices(
        val_dir,
        split["splits"]["val_train"],
        args.sat_qtype_filter,
        "val_train",
    )
    eval_indices = filter_sat_indices(
        val_dir,
        split["splits"]["val_eval"],
        args.sat_qtype_filter,
        "val_eval",
    )
    train_ds = SATMCQDataset(
        split_dir=val_dir,
        split_name="val_train",
        indices=train_indices,
    )
    eval_ds = SATMCQDataset(
        split_dir=val_dir,
        split_name="val_eval",
        indices=eval_indices,
    )
    print(f"Train samples: {len(train_ds)}")
    print(f"Val-eval samples: {len(eval_ds)}")
    return train_ds, eval_ds


def get_trainable_params(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found.")
    return params


@torch.no_grad()
def evaluate(model, loader, max_samples: int = 0):
    model.eval()
    total = correct = 0
    loss_sum = 0.0

    for batch in tqdm(loader, desc="eval", leave=False):
        inputs = model._prepare_inputs(batch["images"], batch["options"], batch.get("question_texts"))
        outputs = model.model(**inputs, return_dict=True, output_hidden_states=False, use_cache=False)
        next_logits = outputs.logits[:, -1, :]
        max_opts = max(len(opts) for opts in batch["options"])
        letter_ids = model.option_token_ids[:max_opts].to(next_logits.device)
        option_logits = next_logits.index_select(dim=-1, index=letter_ids)
        target = torch.tensor(batch["labels"], dtype=torch.long, device=option_logits.device)
        loss_task = F.cross_entropy(option_logits, target)

        bs = len(batch["labels"])
        total += bs
        correct += (option_logits.argmax(dim=-1).cpu() == torch.tensor(batch["labels"])).sum().item()
        loss_sum += loss_task.item() * bs
        if max_samples > 0 and total >= max_samples:
            break

    return {"acc": correct / max(total, 1), "loss": loss_sum / max(total, 1), "n": total}


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
    ax.set_xlabel("step"); ax.set_ylabel("task loss"); ax.set_title("Task Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, [r["align"] for r in train_log], alpha=0.6)
    ax.set_xlabel("step"); ax.set_ylabel("align loss"); ax.set_title("Align Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if eval_log:
        ax.plot([r["step"] for r in eval_log], [r["acc"] for r in eval_log], "o-")
    ax.set_xlabel("step"); ax.set_ylabel("accuracy"); ax.set_title("Eval Accuracy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(out_path, dpi=150); plt.close()


def subset_batch(batch: Dict, indices: List[int]) -> Dict:
    return {
        "images": [batch["images"][i] for i in indices],
        "options": [batch["options"][i] for i in indices],
        "labels": [batch["labels"][i] for i in indices],
        "question_texts": [batch["question_texts"][i] for i in indices],
        "question_types": [batch["question_types"][i] for i in indices],
        "split_names": [batch["split_names"][i] for i in indices],
        "raw": [batch["raw"][i] for i in indices],
    }


def task_only_forward(model: Gemma4LingBotSpatialModel, batch: Dict) -> ForwardOutput:
    inputs = model._prepare_inputs(batch["images"], batch["options"], batch.get("question_texts"))
    outputs = model.model(**inputs, return_dict=True, output_hidden_states=False, use_cache=False)
    next_logits = outputs.logits[:, -1, :]
    max_opts = max(len(opts) for opts in batch["options"])
    letter_ids = model.option_token_ids[:max_opts].to(next_logits.device)
    option_logits = next_logits.index_select(dim=-1, index=letter_ids)
    target = torch.tensor(batch["labels"], dtype=torch.long, device=option_logits.device)
    loss_task = F.cross_entropy(option_logits, target)
    zero = torch.tensor(0.0, device=option_logits.device)
    return ForwardOutput(
        loss=loss_task,
        loss_task=loss_task,
        loss_align=zero,
        loss_preserve=zero,
        logits=option_logits,
        pred_indices=option_logits.argmax(dim=-1),
    )


@contextmanager
def temporary_teacher_mode(model: Gemma4LingBotSpatialModel, teacher_mode: str):
    old_model_mode = model.teacher_mode
    old_teacher_mode = model.lingbot_teacher.teacher_mode
    model.teacher_mode = teacher_mode
    model.lingbot_teacher.teacher_mode = teacher_mode
    try:
        yield
    finally:
        model.teacher_mode = old_model_mode
        model.lingbot_teacher.teacher_mode = old_teacher_mode


def _teacher_prompt_args(model: Gemma4LingBotSpatialModel, question_texts):
    if model.wan_use_prompt:
        return question_texts, None
    return None, None


def double_image_forward(
    model: Gemma4LingBotSpatialModel,
    batch: Dict,
    mode: str,
) -> ForwardOutput:
    if mode not in {"pairwise_single_teacher", "mean_pool"}:
        raise ValueError(f"Unsupported double-image mode: {mode}")

    image_counts = model._image_counts(batch["images"])
    batch_size = len(batch["images"])
    zero = torch.tensor(0.0, device=model.device)

    h0_vis = None
    if model.needs_base and model.preserve_mode == "vision_only":
        base_inputs = model._prepare_inputs(batch["images"], batch["options"], batch.get("question_texts"))
        with torch.no_grad():
            h0_vis = model._extract_base_visual_preserve_features(base_inputs, image_counts).detach()

    model._captured_student_visual = None
    inputs = model._prepare_inputs(batch["images"], batch["options"], batch.get("question_texts"))
    logits, h = model._extract_projected_image_tokens(
        model.model,
        inputs,
        batch_size,
        image_counts=image_counts,
        target_token_len=2,
    )

    next_logits = logits[:, -1, :]
    max_opts = max(len(opts) for opts in batch["options"])
    letter_ids = model.option_token_ids[:max_opts].to(next_logits.device)
    option_logits = next_logits.index_select(dim=-1, index=letter_ids)
    target = torch.tensor(batch["labels"], dtype=torch.long, device=option_logits.device)
    loss_task = F.cross_entropy(option_logits, target)

    z_m_tokens = model.student_align(h)
    wan_prompts, wan_embeds = _teacher_prompt_args(model, batch.get("question_texts"))

    if mode == "pairwise_single_teacher":
        first_images = [sample_images[0] for sample_images in batch["images"]]
        second_images = [sample_images[1] for sample_images in batch["images"]]
        with temporary_teacher_mode(model, "i2v"):
            teacher_feat_1 = model.lingbot_teacher(
                first_images,
                target_token_len=1,
                prompts=wan_prompts,
                prompt_embeddings=wan_embeds,
            ).to(device=model.device, dtype=torch.bfloat16)
            teacher_feat_2 = model.lingbot_teacher(
                second_images,
                target_token_len=1,
                prompts=wan_prompts,
                prompt_embeddings=wan_embeds,
            ).to(device=model.device, dtype=torch.bfloat16)
        teacher_feat = torch.cat([teacher_feat_1, teacher_feat_2], dim=1)
        z_m = z_m_tokens
    else:
        with temporary_teacher_mode(model, "i2v_pair"):
            teacher_feat = model.lingbot_teacher(
                batch["images"],
                target_token_len=1,
                prompts=wan_prompts,
                prompt_embeddings=wan_embeds,
            ).to(device=model.device, dtype=torch.bfloat16)
        z_m = z_m_tokens.mean(dim=1, keepdim=True)

    z_g = model.teacher_align(teacher_feat)
    loss_align = _cosine_align(z_m, z_g.detach())

    if model.needs_base:
        if model.preserve_mode == "vision_only":
            h_vis = model._extract_student_visual_preserve_features(image_counts)
            loss_preserve = _normalize_mse(h_vis, h0_vis.detach())
        else:
            with torch.no_grad():
                base_inputs = model._prepare_inputs(batch["images"], batch["options"], batch.get("question_texts"))
                _, h0 = model._extract_projected_image_tokens(
                    model.base_model,
                    base_inputs,
                    batch_size,
                    image_counts=image_counts,
                    target_token_len=2,
                )
            loss_preserve = _normalize_mse(h, h0.detach())
    else:
        loss_preserve = zero

    total_loss = loss_task + model.lambda_align * loss_align + model.lambda_preserve * loss_preserve
    return ForwardOutput(
        loss=total_loss,
        loss_task=loss_task,
        loss_align=loss_align,
        loss_preserve=loss_preserve,
        logits=option_logits,
        pred_indices=option_logits.argmax(dim=-1),
    )


def mixed_forward(model: Gemma4LingBotSpatialModel, batch: Dict, double_image_align_mode: str) -> ForwardOutput:
    single_idx, multi_idx = [], []
    for i, sample_images in enumerate(batch["images"]):
        if len(sample_images) == 1:
            single_idx.append(i)
        else:
            multi_idx.append(i)

    outputs = []
    counts = []
    if single_idx:
        sb = subset_batch(batch, single_idx)
        outputs.append(model(sb["images"], sb["options"], sb["labels"], question_texts=sb.get("question_texts")))
        counts.append(len(single_idx))
    if multi_idx:
        mb = subset_batch(batch, multi_idx)
        outputs.append(double_image_forward(model, mb, double_image_align_mode))
        counts.append(len(multi_idx))

    total = sum(counts)
    device = outputs[0].loss.device
    loss = torch.tensor(0.0, device=device)
    loss_task = torch.tensor(0.0, device=device)
    loss_align = torch.tensor(0.0, device=device)
    loss_preserve = torch.tensor(0.0, device=device)
    logits_parts = []
    pred_parts = []
    for out, count in zip(outputs, counts):
        w = count / total
        loss = loss + out.loss * w
        loss_task = loss_task + out.loss_task * w
        loss_align = loss_align + out.loss_align * w
        loss_preserve = loss_preserve + out.loss_preserve * w
        logits_parts.append(out.logits)
        pred_parts.append(out.pred_indices)

    logits = torch.cat(logits_parts, dim=0)
    pred_indices = torch.cat(pred_parts, dim=0)
    return ForwardOutput(
        loss=loss,
        loss_task=loss_task,
        loss_align=loss_align,
        loss_preserve=loss_preserve,
        logits=logits,
        pred_indices=pred_indices,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma_model_name", type=str, required=True)
    parser.add_argument("--lingbot_model_dir", type=str, required=True)
    parser.add_argument("--lingbot_code_dir", type=str, required=True)
    parser.add_argument("--sat_root", type=str, default=SAT_ROOT)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--teacher_gpu", type=int, default=1)
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
    parser.add_argument("--lambda_align", type=float, default=0.1)
    parser.add_argument("--lambda_preserve", type=float, default=0.05)
    parser.add_argument("--teacher_mode", type=str, default="i2v")
    parser.add_argument("--i2v_num_frames", type=int, default=9)
    parser.add_argument("--num_teacher_steps", type=int, default=2)
    parser.add_argument("--use_fast_model", action="store_true", default=True)
    parser.add_argument("--wan_hook_block_index", type=int, default=24)
    parser.add_argument("--use_camera_perturbation", action="store_true")
    parser.add_argument("--wan_prompt_text", type=str, default=None)
    parser.add_argument(
        "--sat_qtype_filter",
        type=str,
        default="all",
        choices=["all", "action_sequence", "non_action_sequence"],
        help=(
            "SAT subset filter. action_sequence keeps val action_sequence and "
            "test ego_movement; non_action_sequence excludes those counterparts."
        ),
    )
    parser.add_argument(
        "--double_image_align_mode",
        type=str,
        default="pairwise_single_teacher",
        choices=["pairwise_single_teacher", "mean_pool"],
    )
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
    print(f"Student device: {student_device}, Teacher device: {teacher_device}")

    train_ds, eval_ds = build_datasets(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_sat_mcq, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_sat_mcq, pin_memory=True)

    model = Gemma4LingBotSpatialModel(
        gemma_model_name=args.gemma_model_name,
        lingbot_model_dir=args.lingbot_model_dir,
        lingbot_code_dir=args.lingbot_code_dir,
        align_dim=args.align_dim,
        align_hidden_dim=args.align_hidden_dim,
        lambda_align=args.lambda_align,
        lambda_preserve=args.lambda_preserve,
        teacher_mode=args.teacher_mode,
        i2v_num_frames=args.i2v_num_frames,
        num_teacher_steps=args.num_teacher_steps,
        use_fast_model=args.use_fast_model,
        use_camera_perturbation=args.use_camera_perturbation,
        wan_hook_block_index=args.wan_hook_block_index,
        wan_prompt_text=args.wan_prompt_text,
        student_device=student_device,
        teacher_device=teacher_device,
    )
    model.option_token_ids = model.option_token_ids[:2]

    trainable_params = get_trainable_params(model)
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    history, train_log, eval_log = [], [], []
    total_steps_per_epoch = len(train_loader)
    eval_interval = max(1, total_steps_per_epoch // args.eval_per_epoch)
    eval_max_samples = max(1, len(eval_ds) // 4)
    print(f"Steps/epoch: {total_steps_per_epoch}, eval every {eval_interval} steps (eval samples: {eval_max_samples})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        running = {"loss": 0.0, "task": 0.0, "align": 0.0, "pres": 0.0, "n": 0}

        for step, batch in enumerate(pbar, start=1):
            out = mixed_forward(model, batch, args.double_image_align_mode)
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
                metrics = evaluate(model, eval_loader, max_samples=eval_max_samples)
                metrics["epoch"] = epoch
                metrics["step"] = global_step
                eval_log.append(metrics)
                history.append(metrics)
                print(f"\n[mid-epoch eval] step {global_step}: acc={metrics['acc']:.4f} loss={metrics['loss']:.4f}")
                model.train()

        metrics = evaluate(model, eval_loader, max_samples=eval_max_samples)
        metrics["epoch"] = epoch
        metrics["step"] = global_step
        eval_log.append(metrics)
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        ckpt_dir = os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        trainable_state = {k: v.cpu() for k, v in model.state_dict().items()
                          if k in trainable_names or k.startswith("student_align") or k.startswith("teacher_align")}
        torch.save(trainable_state, os.path.join(ckpt_dir, "trainable_state.pt"))
        saver = getattr(model, "processor", None) or getattr(model, "tokenizer", None)
        if saver is not None:
            try:
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
