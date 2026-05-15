"""
Train Gemma4 + LingBot on EmbSpatial-Bench.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_embspatial_mcq import EmbSpatialMCQDataset, collate_embspatial_mcq
from gemma4_lingbot_spatial_model import Gemma4LingBotSpatialModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_trainable_params(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found.")
    return params


def build_datasets(args):
    split = load_split(args.split_file)
    train_ds = EmbSpatialMCQDataset(args.json_path, indices=split["splits"]["train"])
    test_ds = EmbSpatialMCQDataset(args.json_path, indices=split["splits"]["test"])
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    return train_ds, test_ds


@torch.no_grad()
def evaluate(model, loader, max_samples: int = 0):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for batch in tqdm(loader, desc="eval", leave=False):
        inputs = model._prepare_inputs(batch["images"], batch["options"], batch.get("question_texts"))
        outputs = model.model(**inputs, return_dict=True, output_hidden_states=False, use_cache=False)
        next_logits = outputs.logits[:, -1, :]
        letter_ids = model.option_token_ids[:4].to(next_logits.device)
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
    axes[0, 0].plot(steps, [r["loss"] for r in train_log], alpha=0.6, label="train")
    if eval_log:
        axes[0, 0].plot([r["step"] for r in eval_log], [r["loss"] for r in eval_log], "o-", label="eval")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].set_title("Total Loss")
    axes[0, 1].plot(steps, [r["task"] for r in train_log], alpha=0.6); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_title("Task")
    axes[1, 0].plot(steps, [r["align"] for r in train_log], alpha=0.6); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].set_title("Align")
    axes[1, 1].plot(steps, [r["pres"] for r in train_log], alpha=0.6); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].set_title("Preserve")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma_model_name", type=str, required=True)
    parser.add_argument("--lingbot_model_dir", type=str, required=True)
    parser.add_argument("--lingbot_code_dir", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
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
    parser.add_argument("--async_teacher_overlap", action="store_true")
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

    train_ds, test_ds = build_datasets(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_embspatial_mcq, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_embspatial_mcq, pin_memory=True)

    if args.async_teacher_overlap:
        from gemma4_lingbot_async_model import Gemma4LingBotAsyncModel as ModelCls
    else:
        from gemma4_lingbot_spatial_model import Gemma4LingBotSpatialModel as ModelCls

    model = ModelCls(
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
    model.option_token_ids = model.option_token_ids[:4]
    trainable_params = get_trainable_params(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    train_log, eval_log, history = [], [], []
    total_steps_per_epoch = len(train_loader)
    eval_interval = max(1, total_steps_per_epoch // args.eval_per_epoch)
    eval_max_samples = max(1, len(test_ds) // 10)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        running = {"loss": 0.0, "task": 0.0, "align": 0.0, "pres": 0.0, "n": 0}

        for step, batch in enumerate(pbar, start=1):
            out = model(batch["images"], batch["options"], batch["labels"], question_texts=batch.get("question_texts"))
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
            train_log.append({"step": global_step, "loss": out.loss.item(), "task": out.loss_task.item(), "align": out.loss_align.item(), "pres": out.loss_preserve.item()})
            pbar.set_postfix({k: f"{running[k]/running['n']:.4f}" for k in ["loss", "task", "align", "pres"]})
            if step % eval_interval == 0 and step < total_steps_per_epoch:
                metrics = evaluate(model, test_loader, max_samples=eval_max_samples)
                metrics["epoch"] = epoch; metrics["step"] = global_step
                eval_log.append(metrics); history.append(metrics); model.train()

        metrics = evaluate(model, test_loader, max_samples=eval_max_samples)
        metrics["epoch"] = epoch; metrics["step"] = global_step
        eval_log.append(metrics); history.append(metrics)

        ckpt_dir = os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        trainable_state = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names or k.startswith("student_align") or k.startswith("teacher_align")}
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


if __name__ == "__main__":
    main()
