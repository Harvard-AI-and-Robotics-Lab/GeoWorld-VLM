"""
Gemma4 + LingBot async-overlap model wrapper.

This keeps the original training semantics and overlaps teacher execution on the
teacher GPU with student/preserve work on the student GPU.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image

from gemma4_lingbot_spatial_model import (
    ForwardOutput,
    Gemma4LingBotSpatialModel,
    _cosine_align,
    _flatten_feature_to_tokens,
    _normalize_mse,
    _pool_tokens_to_len,
)


class Gemma4LingBotAsyncModel(Gemma4LingBotSpatialModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingbot_teacher")
        if isinstance(self.teacher_device, str) and self.teacher_device.startswith("cuda"):
            self._teacher_stream = torch.cuda.Stream(device=self.teacher_device)
        else:
            self._teacher_stream = None

    def _compute_teacher_feat_body(
        self,
        images,
        wan_prompts,
        wan_embeds,
    ) -> torch.Tensor:
        if self.dual_timesteps:
            feats = []
            orig_t = self.lingbot_teacher.target_timestep
            for ts in self.dual_timesteps:
                self.lingbot_teacher.target_timestep = ts
                feat = self.lingbot_teacher(
                    images,
                    target_token_len=1,
                    prompts=wan_prompts,
                    prompt_embeddings=wan_embeds,
                )
                feats.append(feat)
            self.lingbot_teacher.target_timestep = orig_t
            return torch.cat(feats, dim=-1)
        if self.dual_blocks:
            self._dual_captured = [None] * len(self.dual_blocks)
            _ = self.lingbot_teacher(
                images,
                target_token_len=1,
                prompts=wan_prompts,
                prompt_embeddings=wan_embeds,
            )
            feats = []
            for cap in self._dual_captured:
                f = _flatten_feature_to_tokens(cap)
                feats.append(_pool_tokens_to_len(f, 1))
            return torch.cat(feats, dim=-1)
        return self.lingbot_teacher(
            images,
            target_token_len=1,
            prompts=wan_prompts,
            prompt_embeddings=wan_embeds,
        )

    def _compute_teacher_feat_sync(
        self,
        images,
        wan_prompts,
        wan_embeds,
    ) -> torch.Tensor:
        if self._teacher_stream is not None:
            with torch.cuda.device(self.teacher_device), torch.cuda.stream(self._teacher_stream):
                feat = self._compute_teacher_feat_body(images, wan_prompts, wan_embeds)
            self._teacher_stream.synchronize()
        else:
            feat = self._compute_teacher_feat_body(images, wan_prompts, wan_embeds)
        return feat.to(device=self.device, dtype=torch.bfloat16)

    def forward(
        self,
        images: Sequence[Union[Image.Image, Sequence[Image.Image]]],
        options_batch: Sequence[Sequence[str]],
        labels: Sequence[int],
        question_texts: Optional[Sequence[str]] = None,
        wan_captions: Optional[Sequence[str]] = None,
        wan_caption_embeddings: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> ForwardOutput:
        image_counts = self._image_counts(images)
        align_token_len = 2 if self.teacher_mode == "i2v_pair" else 1
        zero = torch.tensor(0.0, device=self.device)

        if wan_caption_embeddings and any(e is not None for e in wan_caption_embeddings):
            wan_prompts = None
            wan_embeds = wan_caption_embeddings
        elif wan_captions and any(c for c in wan_captions):
            wan_prompts = wan_captions
            wan_embeds = None
        elif self.wan_use_prompt:
            wan_prompts = question_texts
            wan_embeds = None
        else:
            wan_prompts = None
            wan_embeds = None

        teacher_future = None
        if not self.task_only:
            teacher_future = self._teacher_pool.submit(
                self._compute_teacher_feat_sync,
                images,
                wan_prompts,
                wan_embeds,
            )

        h0_vis = None
        if self.needs_base and self.preserve_mode == "vision_only":
            base_inputs = self._prepare_inputs(images, options_batch, question_texts)
            with torch.no_grad():
                h0_vis = self._extract_base_visual_preserve_features(base_inputs, image_counts).detach()

        self._captured_student_visual = None
        inputs = self._prepare_inputs(images, options_batch, question_texts)
        batch_size = len(images)

        if self.align_only:
            logits, h = self._extract_projected_image_tokens(
                self.model,
                inputs,
                batch_size,
                image_counts=image_counts,
                target_token_len=align_token_len,
            )
            loss_task = zero
            option_logits = torch.zeros(batch_size, 6, device=self.device)
        elif self.task_only:
            outputs = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=False,
                use_cache=False,
            )
            logits = outputs.logits
            next_logits = logits[:, -1, :]
            letter_ids = self.option_token_ids.to(next_logits.device)
            option_logits = next_logits.index_select(dim=-1, index=letter_ids)
            target = torch.tensor(labels, dtype=torch.long, device=option_logits.device)
            loss_task = F.cross_entropy(option_logits, target)
            return ForwardOutput(
                loss=loss_task,
                loss_task=loss_task,
                loss_align=zero,
                loss_preserve=zero,
                logits=option_logits,
                pred_indices=option_logits.argmax(dim=-1),
            )
        else:
            logits, h = self._extract_projected_image_tokens(
                self.model,
                inputs,
                batch_size,
                image_counts=image_counts,
                target_token_len=align_token_len,
            )
            next_logits = logits[:, -1, :]
            letter_ids = self.option_token_ids.to(next_logits.device)
            option_logits = next_logits.index_select(dim=-1, index=letter_ids)
            target = torch.tensor(labels, dtype=torch.long, device=option_logits.device)
            loss_task = F.cross_entropy(option_logits, target)

        if teacher_future is not None:
            teacher_feat = teacher_future.result()
            z_m = self.student_align(h)
            z_g = self.teacher_align(teacher_feat)
            loss_align = _cosine_align(z_m, z_g.detach())
        else:
            loss_align = zero

        if self.needs_base:
            if self.preserve_mode == "vision_only":
                h_vis = self._extract_student_visual_preserve_features(image_counts)
                loss_preserve = _normalize_mse(h_vis, h0_vis.detach())
            else:
                with torch.no_grad():
                    base_inputs = self._prepare_inputs(images, options_batch, question_texts)
                    _, h0 = self._extract_projected_image_tokens(
                        self.base_model,
                        base_inputs,
                        batch_size,
                        image_counts=image_counts,
                        target_token_len=align_token_len,
                    )
                loss_preserve = _normalize_mse(h, h0.detach())
        else:
            loss_preserve = zero

        if self.align_only:
            total_loss = loss_align
        else:
            total_loss = loss_task + self.lambda_align * loss_align + self.lambda_preserve * loss_preserve

        pred_indices = option_logits.argmax(dim=-1)
        return ForwardOutput(
            loss=total_loss,
            loss_task=loss_task,
            loss_align=loss_align,
            loss_preserve=loss_preserve,
            logits=option_logits,
            pred_indices=pred_indices,
        )
