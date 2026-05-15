"""
Gemma4 + LingBot-World spatial alignment model.

GPU 0: Gemma student, frozen base model, and alignment MLPs.
GPU 1: LingBot-World teacher. The text encoder can stay on CPU when prompt
embeddings are cached.
"""
import math
import os
import random
import sys
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor
from hf_compat import patch_all_tied_weights_keys
from internvl_utils import is_internvl_model, load_internvl_tokenizer, prepare_internvl_inputs

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

GEMMA_MODEL_PATH = os.environ.get("GEMMA_MODEL", "models/gemma-4-E4B-it")
LINGBOT_MODEL_PATH = os.environ.get("LINGBOT_MODEL", "models/lingbot-world-base-cam")
LINGBOT_CODE_PATH = os.environ.get("LINGBOT_CODE", "external/lingbot-world")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1, lazy: bool = False):
        super().__init__()
        first = nn.LazyLinear(hidden_dim) if lazy else nn.Linear(in_dim, hidden_dim)
        self.net = nn.Sequential(
            first,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _is_qwen3_vl_model(model_path: str) -> bool:
    return "qwen3-vl" in os.path.basename(os.path.normpath(model_path)).lower()


def _load_image_text_model(model_path: str, dtype: torch.dtype, device: str):
    attempts = [{"attn_implementation": "eager"}]
    if _is_qwen3_vl_model(model_path):
        attempts = [{"attn_implementation": "flash_attention_2"}]
        attempts.append({"use_flash_attn": True})
        attempts.append({"attn_implementation": "eager"})
    last_error = None
    for extra_kwargs in attempts:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=dtype,
                local_files_only=True,
                **extra_kwargs,
            ).to(device)
            print(f"Loaded VLM with kwargs: {extra_kwargs}")
            return model
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to load VLM from {model_path}. Last error: {last_error}")


def _load_internvl_model(model_path: str, dtype: torch.dtype, device: str):
    attempts = [
        {"use_flash_attn": True},
        {"use_flash_attn": False},
    ]
    last_error = None
    for extra_kwargs in attempts:
        try:
            model = AutoModel.from_pretrained(
                model_path,
                dtype=dtype,
                local_files_only=True,
                trust_remote_code=True,
                **extra_kwargs,
            ).to(device)
            print(f"Loaded InternVL with kwargs: {extra_kwargs}")
            return model
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to load InternVL from {model_path}. Last error: {last_error}")


@dataclass
class ForwardOutput:
    loss: torch.Tensor
    loss_task: torch.Tensor
    loss_align: torch.Tensor
    loss_preserve: torch.Tensor
    logits: torch.Tensor
    pred_indices: torch.Tensor


def _first_single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    text2 = text.strip()
    ids = tokenizer.encode(text2, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    raise ValueError(f"Cannot encode `{text}` as single token.")


def _normalize_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return F.mse_loss(x, y)


def _cosine_align(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (1.0 - (x * y).sum(dim=-1)).mean()


def _pool_tokens_to_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[1] == target_len:
        return x
    x = x.transpose(1, 2)
    x = F.adaptive_avg_pool1d(x, target_len)
    x = x.transpose(1, 2)
    return x


def _flatten_feature_to_tokens(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim == 5:
        feat = feat.permute(0, 2, 3, 4, 1).contiguous()
        feat = feat.view(feat.shape[0], -1, feat.shape[-1])
        return feat
    if feat.ndim == 4:
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.shape[0], -1, feat.shape[-1])
        return feat
    if feat.ndim == 3:
        return feat
    if feat.ndim == 2:
        return feat.unsqueeze(1)
    raise ValueError(f"Unsupported feature shape: {tuple(feat.shape)}")


CAMERA_ACTION_POOL = [
    "w", "s", "a", "d",      # single-direction translation
    "wa", "wd", "sa", "sd",  # diagonal translation
    "wj", "wl",              # forward + left/right turn
    "sj", "sl",              # backward + left/right turn
    "i", "k",                # pitch up/down
]


class LingBotWorldTeacher(nn.Module):
    """
    Frozen LingBot-World A14B teacher (on dedicated GPU).
    T5 stays on CPU, blank prompt is cached after first call.
    """

    def __init__(
        self,
        checkpoint_dir: str = None,
        code_dir: str = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 9,
        target_timestep: int = 300,
        shift: float = 5.0,
        hook_block_index: int = -1,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:1",
        teacher_mode: str = "static",
        i2v_num_frames: int = 33,
        i2v_denoise_steps: int = 50,
        jitter_strength: float = 0.3,
        num_teacher_steps: int = 0,
        use_fast_model: bool = False,
        use_camera_perturbation: bool = False,
        wan_prompt_text: str = None,
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir or LINGBOT_MODEL_PATH
        code_dir = code_dir or LINGBOT_CODE_PATH
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.teacher_mode = teacher_mode
        self.jitter_strength = jitter_strength
        self.num_teacher_steps = num_teacher_steps
        self.use_fast_model = use_fast_model
        self.use_camera_perturbation = use_camera_perturbation
        self.wan_prompt_text = wan_prompt_text
        if teacher_mode == "single":
            self.num_frames = 1
        elif teacher_mode in ("i2v", "i2v_pair", "bookend"):
            self.num_frames = i2v_num_frames
        self.i2v_denoise_steps = i2v_denoise_steps
        self.target_timestep = target_timestep
        self.shift = shift
        self.hook_block_index = hook_block_index
        self.device = device
        self.torch_dtype = torch_dtype

        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        from wan.configs.wan_i2v_A14B import i2v_A14B as config
        from wan.modules.model import WanModel
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.modules.t5 import T5EncoderModel

        self.config = config
        self.vae_stride = config.vae_stride  # (4, 8, 8)
        self.patch_size = config.patch_size  # (1, 2, 2)

        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        print(f"Teacher mode: {self.teacher_mode}, num_frames={self.num_frames}")

        # VAE on teacher GPU
        print(f"Loading LingBot VAE on {device}...")
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(self.checkpoint_dir, config.vae_checkpoint),
            device=device,
        )

        # T5 on CPU — we cache the blank prompt so it only runs once
        print(f"Loading T5 encoder on CPU...")
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(self.checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(self.checkpoint_dir, config.t5_tokenizer),
        )
        self._cached_blank_context = {}  # batch_size -> list of tensors on device

        # Transformer on teacher GPU
        subfolder = config.fast_noise_checkpoint if use_fast_model else config.low_noise_checkpoint
        # Fix underscore/hyphen mismatch between config and actual directory name
        if not os.path.exists(os.path.join(self.checkpoint_dir, subfolder)):
            alt = subfolder.replace('_', '-')
            if os.path.exists(os.path.join(self.checkpoint_dir, alt)):
                print(f"Subfolder '{subfolder}' not found, using '{alt}' instead.")
                subfolder = alt
        print(f"Loading LingBot transformer ({subfolder}) on {device}...")
        self.model = WanModel.from_pretrained(
            self.checkpoint_dir,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
            control_type='cam',
        )
        self.model.eval().requires_grad_(False)
        self.model.to(device)
        self.dim = config.dim  # 5120
        print(f"LingBot model loaded! dim={self.dim}, layers={config.num_layers}")

        # Hook middle transformer block
        self._captured_feature: Optional[torch.Tensor] = None
        num_blocks = len(self.model.blocks)
        if hook_block_index < 0:
            self.hook_index = num_blocks // 2
        else:
            self.hook_index = min(hook_block_index, num_blocks - 1)
        self._feature_hook = self.model.blocks[self.hook_index].register_forward_hook(self._save_hook)
        print(f"LingBot hook: blocks[{self.hook_index}] (of {num_blocks})")

        if use_fast_model:
            # Match lingbot-world/wan/image2video_fast.py:
            # timesteps are selected by indexing scheduler.timesteps with
            # [0, 179, 358, 679] after set_timesteps(1000, shift=5.0).
            self.fast_timesteps = [999, 957, 899, 702]
            self.fast_sigmas = [
                0.9997998398718975,
                0.9579927750404370,
                0.8994113476291231,
                0.7024066944814861,
            ]
            print(f"Fast timesteps: {self.fast_timesteps}")
            print(f"Fast sigmas: {self.fast_sigmas}")

        # Camera perturbation imports (lazy — only when needed)
        if use_camera_perturbation:
            from wan.utils.cam_utils import (
                compute_relative_poses,
                interpolate_camera_poses,
                get_plucker_embeddings,
            )
            from wan.utils.wasd_ijkl_to_c2ws import (
                wasd_array_to_frame_keys,
                generate_and_save_trajectory,
            )
            self._cam_compute_relative_poses = compute_relative_poses
            self._cam_interpolate_poses = interpolate_camera_poses
            self._cam_get_plucker = get_plucker_embeddings
            self._cam_wasd_to_keys = wasd_array_to_frame_keys
            self._cam_gen_traj = generate_and_save_trajectory
            print(f"Camera perturbation enabled, action pool: {CAMERA_ACTION_POOL}")

        # Pre-compute prompt embedding if a fixed prompt text is given
        self._cached_prompt_context = None
        if wan_prompt_text:
            print(f"Pre-encoding fixed prompt: '{wan_prompt_text}'")
            prompt_ctx_cpu = self.text_encoder([wan_prompt_text], torch.device('cpu'))
            self._cached_prompt_context = [t.to(self.device) for t in prompt_ctx_cpu]
            print(f"Fixed prompt encoded and cached.")

        # Free T5 GPU memory — keep on CPU only
        self.text_encoder.model.to('cpu')
        torch.cuda.empty_cache()

    def _save_hook(self, module, inputs, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        self._captured_feature = out

    @torch.no_grad()
    def _get_blank_context(self, batch_size: int) -> list:
        if batch_size in self._cached_blank_context:
            return self._cached_blank_context[batch_size]
        # T5 stays on CPU, encode there, then move result to teacher device
        context_cpu = self.text_encoder([""] * batch_size, torch.device('cpu'))
        context = [t.to(self.device) for t in context_cpu]
        self._cached_blank_context[batch_size] = context
        return context

    @torch.no_grad()
    def _build_static_clip(self, images: Sequence[Image.Image]) -> torch.Tensor:
        img_tensors = [self.transform(img.convert("RGB")) for img in images]
        img_tensors = torch.stack(img_tensors, dim=0).to(self.device)
        video = img_tensors.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
        return video

    @torch.no_grad()
    def _build_jitter_clip(self, images: Sequence[Image.Image]) -> torch.Tensor:
        s = self.jitter_strength
        jitter_aug = transforms.Compose([
            transforms.RandomAffine(
                degrees=s * 3.0,
                translate=(s * 0.05, s * 0.05),
                scale=(1 - s * 0.05, 1 + s * 0.05),
            ),
            transforms.RandomPerspective(distortion_scale=s * 0.03, p=1.0),
        ])
        frames_list = []
        for img in images:
            rgb = img.convert("RGB")
            per_image_frames = [self.transform(rgb)]
            for _ in range(self.num_frames - 1):
                per_image_frames.append(self.transform(jitter_aug(rgb)))
            frames_list.append(torch.stack(per_image_frames, dim=1))  # [C, T, H, W]
        video = torch.stack(frames_list, dim=0).to(self.device)  # [B, C, T, H, W]
        return video

    @torch.no_grad()
    def _build_noisy_copy_clip(self, images: Sequence[Image.Image]) -> torch.Tensor:
        img_tensors = [self.transform(img.convert("RGB")) for img in images]
        img_tensors = torch.stack(img_tensors, dim=0).to(self.device)  # [B, C, H, W]
        B, C, H, W = img_tensors.shape
        video = img_tensors.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)  # [B, C, T, H, W]
        noise = torch.randn(B, C, self.num_frames - 1, H, W, device=self.device, dtype=video.dtype)
        video[:, :, 1:, :, :] = video[:, :, 1:, :, :] + 0.3 * noise
        video[:, :, 1:, :, :] = video[:, :, 1:, :, :].clamp(-1.0, 1.0)
        return video

    @torch.no_grad()
    def _build_single_frame(self, images: Sequence[Image.Image]) -> torch.Tensor:
        img_tensors = [self.transform(img.convert("RGB")) for img in images]
        img_tensors = torch.stack(img_tensors, dim=0).to(self.device)
        return img_tensors.unsqueeze(2)  # [B, C, 1, H, W]

    @torch.no_grad()
    def _build_i2v_clip(self, images: Sequence[Union[Image.Image, Sequence[Image.Image]]]) -> torch.Tensor:
        img_tensors = []
        for sample in images:
            if isinstance(sample, Image.Image):
                sample_images = [sample]
            else:
                sample_images = list(sample)
            img_tensors.append(self.transform(sample_images[0].convert("RGB")))
        img_tensors = torch.stack(img_tensors, dim=0).to(self.device)
        B, C, H, W = img_tensors.shape
        video = torch.zeros(B, C, self.num_frames, H, W, device=self.device, dtype=img_tensors.dtype)
        video[:, :, 0, :, :] = img_tensors
        return video

    @torch.no_grad()
    def _build_i2v_pair_clip(self, images: Sequence[Union[Image.Image, Sequence[Image.Image]]]) -> torch.Tensor:
        sample_videos = []
        for sample in images:
            if isinstance(sample, Image.Image):
                sample_images = [sample]
            else:
                sample_images = list(sample)
            first = self.transform(sample_images[0].convert("RGB")).to(self.device)
            second = self.transform(sample_images[1].convert("RGB")).to(self.device) if len(sample_images) > 1 else first
            C, H, W = first.shape
            video = torch.zeros(C, self.num_frames, H, W, device=self.device, dtype=first.dtype)
            video[:, 0, :, :] = first
            video[:, 1, :, :] = second
            sample_videos.append(video)
        return torch.stack(sample_videos, dim=0)

    @torch.no_grad()
    def _build_bookend_clip(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """First and last frames are original images, middle frames are zeros."""
        img_tensors = [self.transform(img.convert("RGB")) for img in images]
        img_tensors = torch.stack(img_tensors, dim=0).to(self.device)
        B, C, H, W = img_tensors.shape
        video = torch.zeros(B, C, self.num_frames, H, W, device=self.device, dtype=img_tensors.dtype)
        video[:, :, 0, :, :] = img_tensors
        video[:, :, -1, :, :] = img_tensors
        return video

    @torch.no_grad()
    def _encode_video_latents(self, video: torch.Tensor) -> torch.Tensor:
        B = video.shape[0]
        video_list = [video[i] for i in range(B)]
        latents_list = self.vae.encode(video_list)
        latents = torch.stack(latents_list)
        return latents.to(self.torch_dtype)

    @torch.no_grad()
    def _make_noisy_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma = self.target_timestep / 1000.0
        noise = torch.randn_like(latents)
        noisy_latents = (1 - sigma) * latents + sigma * noise
        t_batch = torch.full((latents.shape[0],), self.target_timestep,
                             device=self.device, dtype=torch.long)
        return noisy_latents, t_batch

    @torch.no_grad()
    def _add_noise_at_sigma(self, latents: torch.Tensor, sigma: float) -> torch.Tensor:
        noise = torch.randn_like(latents)
        return ((1 - sigma) * latents + sigma * noise).to(self.torch_dtype)

    @torch.no_grad()
    def _denoise_i2v(self, latents: torch.Tensor, context: list,
                     y_list: list, max_seq_len: int,
                     dit_cond_dict: dict = None) -> None:
        num_steps = self.i2v_denoise_steps
        # Flow-matching schedule: linearly spaced sigmas from 1.0 to 0.0
        # Apply shift to match Wan's shifted schedule
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # Find the step closest to target_timestep
        target_sigma = self.target_timestep / 1000.0
        stop_idx = torch.argmin(torch.abs(sigmas[:-1] - target_sigma)).item()

        B = latents.shape[0]
        noise = torch.randn_like(latents)
        x_t = noise.to(self.torch_dtype)  # start from pure noise

        for i in range(stop_idx + 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            t_val = int(sigma.item() * 1000)
            t_batch = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            _, _, F_lat, H_lat, W_lat = latents.shape
            seq_len = F_lat * H_lat * W_lat // (self.patch_size[1] * self.patch_size[2])
            t_2d = t_batch.unsqueeze(1).expand(-1, seq_len)

            self._captured_feature = None
            x_list = [x_t[j] for j in range(B)]
            with torch.amp.autocast('cuda', dtype=self.torch_dtype):
                v_pred = self.model(
                    x=x_list,
                    t=t_2d,
                    context=context,
                    seq_len=seq_len,
                    y=y_list,
                    dit_cond_dict=dit_cond_dict,
                )
            if isinstance(v_pred, list):
                v_pred = torch.stack(v_pred)
            # Euler step: x_{t+1} = x_t + (sigma_next - sigma) * v_pred
            x_t = x_t + (sigma_next - sigma) * v_pred

    @torch.no_grad()
    def _multi_step_extract(self, noisy_latents: torch.Tensor, context: list,
                            y_list: list, max_seq_len: int,
                            dit_cond_dict: dict = None) -> None:
        B = noisy_latents.shape[0]
        if self.use_fast_model:
            sigmas = torch.tensor(self.fast_sigmas[:self.num_teacher_steps + 1],
                                  device=self.device)
        else:
            target_sigma = self.target_timestep / 1000.0
            sigmas = torch.linspace(target_sigma, 0.0, self.num_teacher_steps + 1,
                                    device=self.device)
        x_t = noisy_latents.to(self.torch_dtype)
        for i in range(self.num_teacher_steps):
            sigma = sigmas[i]
            t_val = int(sigma.item() * 1000)
            t_batch = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            t_2d = t_batch.unsqueeze(1).expand(-1, max_seq_len)
            self._captured_feature = None
            x_list = [x_t[j] for j in range(B)]
            with torch.amp.autocast('cuda', dtype=self.torch_dtype):
                v_pred = self.model(
                    x=x_list, t=t_2d, context=context,
                    seq_len=max_seq_len, y=y_list, dit_cond_dict=dit_cond_dict,
                )
            if i < self.num_teacher_steps - 1:
                if isinstance(v_pred, list):
                    v_pred = torch.stack(v_pred)
                sigma_next = sigmas[i + 1]
                x_t = x_t + (sigma_next - sigma) * v_pred

    @torch.no_grad()
    def _fast_feature_forward(self, latents: torch.Tensor, timestep: int,
                              context: list, y_list: list, max_seq_len: int,
                              dit_cond_dict: dict = None) -> torch.Tensor:
        B = latents.shape[0]
        t_batch = torch.full((B,), timestep, device=self.device, dtype=torch.long)
        t_2d = t_batch.unsqueeze(1).expand(-1, max_seq_len)
        self._captured_feature = None
        x_list = [latents[i] for i in range(B)]
        with torch.amp.autocast('cuda', dtype=self.torch_dtype):
            v_pred = self.model(
                x=x_list,
                t=t_2d,
                context=context,
                seq_len=max_seq_len,
                y=y_list,
                dit_cond_dict=dit_cond_dict,
            )
        if isinstance(v_pred, list):
            v_pred = torch.stack(v_pred)
        return v_pred

    @torch.no_grad()
    def _extract_fast_static_feature(self, latents: torch.Tensor, context: list,
                                     y_list: list, max_seq_len: int,
                                     dit_cond_dict: dict = None) -> None:
        if self.num_teacher_steps < 0 or self.num_teacher_steps >= len(self.fast_sigmas):
            raise ValueError(
                f"Fast static extraction supports num_teacher_steps in [0, {len(self.fast_sigmas) - 1}], "
                f"got {self.num_teacher_steps}."
            )
        sigma = self.fast_sigmas[self.num_teacher_steps]
        timestep = self.fast_timesteps[self.num_teacher_steps]
        noisy_latents = self._add_noise_at_sigma(latents, sigma)
        _ = self._fast_feature_forward(
            noisy_latents,
            timestep,
            context,
            y_list,
            max_seq_len,
            dit_cond_dict=dit_cond_dict,
        )

    @torch.no_grad()
    def _extract_fast_i2v_feature(self, latents: torch.Tensor, context: list,
                                  y_list: list, max_seq_len: int,
                                  dit_cond_dict: dict = None) -> None:
        max_updates = len(self.fast_sigmas) - 1
        if self.num_teacher_steps < 0 or self.num_teacher_steps > max_updates:
            raise ValueError(
                f"Fast i2v extraction supports num_teacher_steps in [0, {max_updates}], "
                f"got {self.num_teacher_steps}."
            )

        x_t = torch.randn_like(latents).to(self.torch_dtype)
        for i in range(self.num_teacher_steps):
            v_pred = self._fast_feature_forward(
                x_t,
                self.fast_timesteps[i],
                context,
                y_list,
                max_seq_len,
                dit_cond_dict=dit_cond_dict,
            )
            sigma = self.fast_sigmas[i]
            sigma_next = self.fast_sigmas[i + 1]
            x_t = x_t + (sigma_next - sigma) * v_pred

        _ = self._fast_feature_forward(
            x_t,
            self.fast_timesteps[self.num_teacher_steps],
            context,
            y_list,
            max_seq_len,
            dit_cond_dict=dit_cond_dict,
        )

    @torch.no_grad()
    def _build_y_conditioning(self, vae_latents: torch.Tensor) -> list:
        B, C, F_lat, H_lat, W_lat = vae_latents.shape
        F_full = self.num_frames

        y_list = []
        for i in range(B):
            if self.teacher_mode == "bookend":
                # Both first and last frames are given
                msk = torch.zeros(1, F_full, H_lat, W_lat, device=self.device)
                msk[:, 0] = 1
                msk[:, -1] = 1
            elif self.teacher_mode == "i2v_pair":
                msk = torch.ones(1, F_full, H_lat, W_lat, device=self.device)
                msk[:, 2:] = 0
            else:
                # Only first frame is given (standard i2v / static)
                msk = torch.ones(1, F_full, H_lat, W_lat, device=self.device)
                msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
            ], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, H_lat, W_lat)
            msk = msk.transpose(1, 2)[0]  # [4, F_lat, H_lat, W_lat]
            y = torch.cat([msk, vae_latents[i]], dim=0).to(self.torch_dtype)
            y_list.append(y)
        return y_list

    @torch.no_grad()
    def _generate_camera_conditioning(self, latents: torch.Tensor) -> dict:
        """Generate random camera conditioning via WASD trajectories."""
        _, _, F_lat, H_lat, W_lat = latents.shape
        # Pick a random action from the pool
        action_combo = random.choice(CAMERA_ACTION_POOL)
        # Build per-frame action arrays for num_frames pixel frames
        n = self.num_frames
        wasd = np.zeros((n, 4), dtype=np.float32)
        ijkl = np.zeros((n, 4), dtype=np.float32)
        wasd_idx = {"w": 0, "a": 1, "s": 2, "d": 3}
        ijkl_idx = {"i": 0, "j": 1, "k": 2, "l": 3}
        for c in action_combo:
            if c in wasd_idx:
                wasd[:, wasd_idx[c]] = 1.0
            elif c in ijkl_idx:
                ijkl[:, ijkl_idx[c]] = 1.0
        # Generate c2w trajectory
        frame_keys = self._cam_wasd_to_keys(wasd, ijkl)
        c2ws_list = self._cam_gen_traj(frame_keys)  # len = n + 1 (includes initial)
        c2ws = np.array(c2ws_list)  # [n+1, 4, 4]

        # Interpolate to latent frame count
        len_c2ws = len(c2ws)
        c2ws_infer = self._cam_interpolate_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, F_lat),
        )
        c2ws_infer = self._cam_compute_relative_poses(c2ws_infer, framewise=True)

        # Default intrinsics for 480x832
        h, w = self.height, self.width
        Ks = torch.tensor([[float(w), float(w), w / 2.0, h / 2.0]])  # [1, 4]
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(self.device)
        Ks = Ks.to(self.device)
        c2ws_plucker_emb = self._cam_get_plucker(c2ws_infer, Ks, h, w)
        # c2ws_plucker_emb: [F_lat, H, W, 6]
        from einops import rearrange
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // H_lat),
            c2=int(w // W_lat),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [1, f*h*w, c]
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'b (f h w) c -> b c f h w',
            f=F_lat, h=H_lat, w=W_lat,
        ).to(self.torch_dtype)
        return {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

    @torch.no_grad()
    def _encode_prompts(self, prompts: Sequence[str]) -> list:
        context_cpu = self.text_encoder(list(prompts), torch.device('cpu'))
        return [t.to(self.device) for t in context_cpu]

    @torch.no_grad()
    def forward(self, images: Sequence[Union[Image.Image, Sequence[Image.Image]]], target_token_len: int,
                prompts: Optional[Sequence[str]] = None,
                prompt_embeddings: Optional[Sequence[Optional[torch.Tensor]]] = None) -> torch.Tensor:
        """Returns features on teacher device. Caller moves to student device."""
        if self.teacher_mode == "single":
            video = self._build_single_frame(images)
        elif self.teacher_mode == "i2v":
            video = self._build_i2v_clip(images)
        elif self.teacher_mode == "i2v_pair":
            video = self._build_i2v_pair_clip(images)
        elif self.teacher_mode == "bookend":
            video = self._build_bookend_clip(images)
        elif self.teacher_mode == "jitter":
            video = self._build_jitter_clip(images)
        elif self.teacher_mode == "noisy_copy":
            video = self._build_noisy_copy_clip(images)
        else:
            video = self._build_static_clip(images)
        latents = self._encode_video_latents(video)

        B = len(images)
        # Prompt priority: embeddings > explicit prompts > fixed prompt text > blank
        if prompt_embeddings is not None and all(e is not None for e in prompt_embeddings):
            context = [e.to(device=self.device, dtype=self.torch_dtype) for e in prompt_embeddings]
        elif prompts is not None:
            context = self._encode_prompts(prompts)
        elif self._cached_prompt_context is not None:
            # Fixed prompt text (e.g. "move far away"), tile to batch size
            context = [self._cached_prompt_context[0]] * B
        else:
            context = self._get_blank_context(B)
        y_list = self._build_y_conditioning(latents)

        _, _, F_lat, H_lat, W_lat = latents.shape
        max_seq_len = F_lat * H_lat * W_lat // (self.patch_size[1] * self.patch_size[2])

        # Camera conditioning
        dit_cond_dict = None
        if self.use_camera_perturbation:
            dit_cond_dict = self._generate_camera_conditioning(latents)

        if self.use_fast_model and self.teacher_mode in ("i2v", "i2v_pair") and self.num_teacher_steps >= 0:
            self._extract_fast_i2v_feature(
                latents, context, y_list, max_seq_len, dit_cond_dict=dit_cond_dict
            )
        elif self.use_fast_model and self.teacher_mode != "i2v" and self.num_teacher_steps >= 0:
            self._extract_fast_static_feature(
                latents, context, y_list, max_seq_len, dit_cond_dict=dit_cond_dict
            )
        elif self.num_teacher_steps > 1:
            noisy_latents, _ = self._make_noisy_latents(latents)
            self._multi_step_extract(noisy_latents, context, y_list, max_seq_len,
                                     dit_cond_dict=dit_cond_dict)
        elif self.num_teacher_steps == 0 and self.teacher_mode in ("i2v", "i2v_pair"):
            self._denoise_i2v(latents, context, y_list, max_seq_len,
                              dit_cond_dict=dit_cond_dict)
        else:
            noisy_latents, t_batch = self._make_noisy_latents(latents)
            self._captured_feature = None
            noisy_latents = noisy_latents.to(self.torch_dtype)
            x_list = [noisy_latents[i] for i in range(B)]
            t_2d = t_batch.unsqueeze(1).expand(-1, max_seq_len)
            with torch.amp.autocast('cuda', dtype=self.torch_dtype):
                _ = self.model(
                    x=x_list,
                    t=t_2d,
                    context=context,
                    seq_len=max_seq_len,
                    y=y_list,
                    dit_cond_dict=dit_cond_dict,
                )

        if self._captured_feature is None:
            raise RuntimeError("LingBot hook did not capture features.")

        feat = _flatten_feature_to_tokens(self._captured_feature)
        feat = _pool_tokens_to_len(feat, target_token_len)
        return feat


class Gemma4LingBotSpatialModel(nn.Module):
    @staticmethod
    def _get_hidden_size(config) -> int:
        if hasattr(config, "llm_config") and hasattr(config.llm_config, "hidden_size"):
            return config.llm_config.hidden_size
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return config.text_config.hidden_size
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        raise AttributeError("Cannot determine hidden_size from model config.")

    def _configure_trainable_params(
        self,
        freeze_llm: bool,
        unfreeze_vision: bool,
        unfreeze_projector: bool,
    ) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

        if not freeze_llm:
            for p in self.model.parameters():
                p.requires_grad = True
            total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable configuration: full model ({total:,} params)")
            return

        vision_keywords = [
            "vision", "visual", "image_encoder", "vision_tower", "image_tower",
            "image_model", "vision_model", "img",
        ]
        projector_keywords = [
            "project", "projector", "projection", "image_projection",
            "multi_modal", "multimodal", "mm_", "mmprojector", "merger", "mlp1",
        ]

        matched_names = []
        for name, p in self.model.named_parameters():
            lname = name.lower()
            is_vision = unfreeze_vision and any(k in lname for k in vision_keywords)
            is_projector = unfreeze_projector and any(k in lname for k in projector_keywords)
            if is_vision or is_projector:
                p.requires_grad = True
                matched_names.append(name)

        if not matched_names and self.task_only:
            fallback_keywords = vision_keywords + projector_keywords + ["connector", "adapter", "resampler"]
            for name, p in self.model.named_parameters():
                lname = name.lower()
                if any(k in lname for k in fallback_keywords):
                    p.requires_grad = True
                    matched_names.append(name)

        trainable = [(n, p.numel()) for n, p in self.model.named_parameters() if p.requires_grad]
        total = sum(n for _, n in trainable)
        print(f"Trainable parameter tensors: {len(trainable)}")
        print(f"Trainable parameter count: {total:,}")
        if trainable:
            preview = ", ".join(name for name, _ in trainable[:8])
            print(f"Trainable name preview: {preview}")

    def __init__(
        self,
        gemma_model_name: str = None,
        lingbot_model_dir: str = None,
        lingbot_code_dir: str = None,
        align_dim: int = 512,
        align_hidden_dim: int = 1024,
        lambda_align: float = 0.2,
        lambda_preserve: float = 0.05,
        freeze_llm: bool = True,
        unfreeze_vision: bool = True,
        unfreeze_projector: bool = True,
        wan_height: int = 480,
        wan_width: int = 832,
        wan_num_frames: int = 9,
        wan_target_timestep: int = 300,
        wan_shift: float = 5.0,
        wan_hook_block_index: int = -1,
        teacher_mode: str = "static",
        i2v_num_frames: int = 33,
        i2v_denoise_steps: int = 50,
        dual_timesteps: list = None,
        dual_blocks: list = None,
        task_only: bool = False,
        align_only: bool = False,
        wan_use_prompt: bool = False,
        jitter_strength: float = 0.3,
        num_teacher_steps: int = 0,
        use_fast_model: bool = False,
        use_camera_perturbation: bool = False,
        wan_prompt_text: str = None,
        student_device: str = "cuda:0",
        teacher_device: str = "cuda:1",
        internvl_max_num: int = 6,
    ):
        super().__init__()
        self.device = student_device
        self.teacher_device = teacher_device
        self.lambda_align = lambda_align
        self.lambda_preserve = lambda_preserve
        self.task_only = task_only
        self.align_only = align_only
        self.wan_use_prompt = wan_use_prompt
        self.needs_teacher = not task_only
        self.teacher_mode = teacher_mode
        self.dual_timesteps = dual_timesteps
        self.dual_blocks = dual_blocks
        self.internvl_max_num = internvl_max_num

        gemma_path = gemma_model_name or GEMMA_MODEL_PATH
        self.is_internvl = is_internvl_model(gemma_path)

        print(f"Loading Gemma model from {gemma_path} on {student_device}...")
        mode = "task_only" if task_only else ("align_only" if align_only else "full")
        print(f"Mode: {mode}")
        if self.is_internvl:
            patch_all_tied_weights_keys()
            self.tokenizer = load_internvl_tokenizer(gemma_path)
            self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            self.model = _load_internvl_model(
                gemma_path,
                dtype=torch.bfloat16,
                device=student_device,
            )
            self.model.img_context_token_id = self.img_context_token_id
        else:
            self.processor = AutoProcessor.from_pretrained(gemma_path, local_files_only=True)
            self.processor.tokenizer.padding_side = "left"
            self.model = _load_image_text_model(
                gemma_path,
                dtype=torch.bfloat16,
                device=student_device,
            )
        self._captured_student_visual = None

        self.preserve_mode = None
        self.needs_base = (not task_only) and (not align_only) and (lambda_preserve > 0)
        if self.needs_base:
            if (not self.is_internvl) and hasattr(self.model, "model") and hasattr(self.model.model, "vision_tower"):
                self.preserve_mode = "vision_only"
                self.base_vision_tower = copy.deepcopy(self.model.model.vision_tower).to(student_device)
                self.base_vision_tower.eval()
                for p in self.base_vision_tower.parameters():
                    p.requires_grad = False
                self._student_vision_hook = self.model.model.vision_tower.register_forward_hook(
                    self._capture_student_visual_hook
                )
            elif self.is_internvl:
                self.base_model = _load_internvl_model(
                    gemma_path,
                    dtype=torch.bfloat16,
                    device=student_device,
                )
                self.base_model.img_context_token_id = self.img_context_token_id
            else:
                self.base_model = _load_image_text_model(
                    gemma_path,
                    dtype=torch.bfloat16,
                    device=student_device,
                )
            if hasattr(self, "base_model"):
                self.base_model.eval()
                for p in self.base_model.parameters():
                    p.requires_grad = False

        print("Gemma model loaded!")
        self._configure_trainable_params(
            freeze_llm=freeze_llm,
            unfreeze_vision=unfreeze_vision,
            unfreeze_projector=unfreeze_projector,
        )

        hidden_size = self._get_hidden_size(self.model.config)

        tokenizer = self.tokenizer if self.is_internvl else self.processor.tokenizer
        self.option_token_ids = self._build_option_token_ids(tokenizer, num_options=6)

        if self.needs_teacher:
            self.student_align = TwoLayerMLP(hidden_size, align_hidden_dim, align_dim).to(torch.bfloat16).to(student_device)

            self.lingbot_teacher = LingBotWorldTeacher(
                checkpoint_dir=lingbot_model_dir,
                code_dir=lingbot_code_dir,
                height=wan_height,
                width=wan_width,
                num_frames=wan_num_frames,
                target_timestep=wan_target_timestep,
                shift=wan_shift,
                hook_block_index=wan_hook_block_index,
                device=teacher_device,
                teacher_mode=teacher_mode,
                i2v_num_frames=i2v_num_frames,
                i2v_denoise_steps=i2v_denoise_steps,
                jitter_strength=jitter_strength,
                num_teacher_steps=num_teacher_steps,
                use_fast_model=use_fast_model,
                use_camera_perturbation=use_camera_perturbation,
                wan_prompt_text=wan_prompt_text,
            )

            teacher_dim = self.lingbot_teacher.dim  # 5120
            if self.dual_timesteps:
                teacher_dim = teacher_dim * len(self.dual_timesteps)
            if self.dual_blocks:
                self._dual_captured = [None] * len(self.dual_blocks)
                self._dual_hooks = []
                blocks = self.lingbot_teacher.model.blocks
                for i, bidx in enumerate(self.dual_blocks):
                    bidx = min(bidx, len(blocks) - 1)
                    def make_hook(idx):
                        def hook_fn(module, inputs, output):
                            self._dual_captured[idx] = output[0] if isinstance(output, tuple) else output
                        return hook_fn
                    self._dual_hooks.append(blocks[bidx].register_forward_hook(make_hook(i)))
                print(f"Dual block hooks: blocks[{self.dual_blocks}]")
                teacher_dim = self.lingbot_teacher.dim * len(self.dual_blocks)
            self.teacher_align = TwoLayerMLP(teacher_dim, align_hidden_dim, align_dim).to(torch.bfloat16).to(student_device)

    def _build_option_token_ids(self, tokenizer, num_options: int = 6) -> torch.Tensor:
        token_ids = []
        for i in range(num_options):
            letter = LETTERS[i]
            try:
                token_id = _first_single_token_id(tokenizer, " " + letter)
            except Exception:
                token_id = _first_single_token_id(tokenizer, letter)
            token_ids.append(token_id)
        return torch.tensor(token_ids, dtype=torch.long)

    def _build_prompt_text(self, options: Sequence[str], question: Optional[str] = None) -> str:
        if question:
            header = question
        else:
            header = "Choose the single caption that best describes the spatial relation in the image."
        lines = [header, "Reply with only one letter."]
        for idx, text in enumerate(options):
            lines.append(f"{LETTERS[idx]}. {text}")
        lines.append("Answer:")
        return "\n".join(lines)

    def _build_messages(self, images: Sequence[Union[Image.Image, Sequence[Image.Image]]], options_batch: Sequence[Sequence[str]],
                        question_texts: Optional[Sequence[str]] = None) -> List[List[Dict]]:
        messages = []
        for i, (img_group, options) in enumerate(zip(images, options_batch)):
            q = question_texts[i] if question_texts and question_texts[i] else None
            prompt_text = self._build_prompt_text(options, question=q)
            if isinstance(img_group, Image.Image):
                sample_images = [img_group]
            else:
                sample_images = list(img_group)
            content = [{"type": "image", "image": img} for img in sample_images]
            content.append({"type": "text", "text": prompt_text})
            messages.append([
                {
                    "role": "user",
                    "content": content,
                }
            ])
        return messages

    def _prepare_inputs(self, images: Sequence[Union[Image.Image, Sequence[Image.Image]]], options_batch: Sequence[Sequence[str]],
                        question_texts: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        if self.is_internvl:
            prompts = []
            for i, options in enumerate(options_batch):
                q = question_texts[i] if question_texts and question_texts[i] else None
                prompts.append(self._build_prompt_text(options, question=q))
            return prepare_internvl_inputs(
                self.model, self.tokenizer, images, prompts, self.device, max_num=self.internvl_max_num
            )

        messages = self._build_messages(images, options_batch, question_texts)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "return_tensors": "pt",
                "padding": True,
            },
        )
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    def _aggregate_per_image_features(
        self,
        per_image_feat: torch.Tensor,
        image_counts: Sequence[int],
        target_token_len: int,
    ) -> torch.Tensor:
        if target_token_len <= 1:
            pooled = []
            idx = 0
            for count in image_counts:
                sample_feat = per_image_feat[idx: idx + count]
                idx += count
                pooled.append(sample_feat.mean(dim=0))
            return torch.stack(pooled, dim=0).unsqueeze(1)
        return self._pack_per_image_features(per_image_feat, image_counts)

    def _extract_projected_image_tokens(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        batch_size: int,
        image_counts: Optional[Sequence[int]] = None,
        target_token_len: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        image_tokens = getattr(outputs, "image_hidden_states", None)

        if isinstance(image_tokens, (tuple, list)):
            image_tokens = image_tokens[-1]

        if image_tokens is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            input_ids = inputs.get("input_ids")
            candidate_ids = []
            for attr in [
                "image_token_id",
                "vision_token_id",
                "image_pad_token_id",
                "vision_pad_token_id",
            ]:
                value = getattr(model.config, attr, None)
                if value is not None:
                    candidate_ids.append(int(value))
            model_token_id = getattr(model, "img_context_token_id", None)
            if model_token_id is not None:
                candidate_ids.append(int(model_token_id))
            vision_config = getattr(model.config, "vision_config", None)
            if vision_config is not None:
                for attr in ["image_token_id", "vision_token_id"]:
                    value = getattr(vision_config, attr, None)
                    if value is not None:
                        candidate_ids.append(int(value))

            if hidden_states is None or not hidden_states or input_ids is None or not candidate_ids:
                raise RuntimeError("No image_hidden_states in model output, and could not infer image token positions.")

            last_hidden = hidden_states[-1]
            pooled = []
            for i in range(batch_size):
                mask = torch.zeros_like(input_ids[i], dtype=torch.bool)
                for token_id in candidate_ids:
                    mask |= (input_ids[i] == token_id)
                sample_tokens = last_hidden[i][mask]
                if sample_tokens.numel() == 0:
                    raise RuntimeError(
                        f"No image token positions found for sample {i}. Candidate token ids: {sorted(set(candidate_ids))}"
                    )
                pooled.append(sample_tokens.mean(dim=0))
            image_tokens = torch.stack(pooled, dim=0).unsqueeze(1)
            if target_token_len > 1:
                image_tokens = image_tokens.expand(-1, target_token_len, -1)
            return outputs.logits, image_tokens

        D = image_tokens.shape[-1]

        if image_tokens.ndim == 4:
            if image_tokens.shape[0] == batch_size:
                per_image_feat = image_tokens.mean(dim=2)
                if target_token_len <= 1:
                    image_tokens = per_image_feat.mean(dim=1, keepdim=True)
                else:
                    if per_image_feat.shape[1] == 1:
                        per_image_feat = per_image_feat.expand(-1, target_token_len, -1)
                    elif per_image_feat.shape[1] >= target_token_len:
                        per_image_feat = per_image_feat[:, :target_token_len, :]
                    else:
                        pad = per_image_feat[:, -1:, :].expand(-1, target_token_len - per_image_feat.shape[1], -1)
                        per_image_feat = torch.cat([per_image_feat, pad], dim=1)
                    image_tokens = per_image_feat
                return outputs.logits, image_tokens
            image_tokens = image_tokens.view(-1, image_tokens.shape[-2], D)

        if image_tokens.ndim == 3:
            if image_tokens.shape[0] == batch_size:
                image_tokens = image_tokens.mean(dim=1, keepdim=True)
                if target_token_len > 1:
                    image_tokens = image_tokens.expand(-1, target_token_len, -1)
            else:
                per_image_feat = image_tokens.mean(dim=1)
                if image_counts is not None:
                    image_tokens = self._aggregate_per_image_features(
                        per_image_feat, image_counts, target_token_len
                    )
                else:
                    image_tokens = per_image_feat.view(batch_size, -1, D).mean(dim=1, keepdim=True)
                    if target_token_len > 1:
                        image_tokens = image_tokens.expand(-1, target_token_len, -1)
                return outputs.logits, image_tokens

        if image_tokens.ndim == 2:
            if image_tokens.shape[0] == batch_size:
                image_tokens = image_tokens.unsqueeze(1)
                if target_token_len > 1:
                    image_tokens = image_tokens.expand(-1, target_token_len, -1)
            else:
                if image_counts is not None:
                    image_tokens = self._aggregate_per_image_features(
                        image_tokens, image_counts, target_token_len
                    )
                else:
                    total = image_tokens.shape[0]
                    tpi = math.ceil(total / batch_size)
                    if total < tpi * batch_size:
                        pad = image_tokens.new_zeros(tpi * batch_size - total, D)
                        image_tokens = torch.cat([image_tokens, pad], dim=0)
                    image_tokens = image_tokens.view(batch_size, tpi, D).mean(dim=1, keepdim=True)
                    if target_token_len > 1:
                        image_tokens = image_tokens.expand(-1, target_token_len, -1)

        return outputs.logits, image_tokens

    def _capture_student_visual_hook(self, module, inputs, output):
        feat = getattr(output, "pooler_output", None)
        if feat is None:
            feat = getattr(output, "last_hidden_state", None)
        if feat is None:
            feat = output
        self._captured_student_visual = feat

    def _image_counts(self, images: Sequence[Union[Image.Image, Sequence[Image.Image]]]) -> List[int]:
        counts = []
        for sample in images:
            if isinstance(sample, Image.Image):
                counts.append(1)
            else:
                counts.append(len(list(sample)))
        return counts

    def _pack_per_image_features(
        self,
        per_image_feat: torch.Tensor,
        image_counts: Sequence[int],
    ) -> torch.Tensor:
        packed = []
        idx = 0
        for count in image_counts:
            sample_feat = per_image_feat[idx: idx + count]
            idx += count
            if sample_feat.shape[0] == 1:
                sample_feat = torch.cat([sample_feat, sample_feat], dim=0)
            else:
                sample_feat = sample_feat[:2]
            packed.append(sample_feat)
        return torch.stack(packed, dim=0)

    def _extract_student_visual_preserve_features(
        self,
        image_counts: Sequence[int],
    ) -> torch.Tensor:
        if self._captured_student_visual is None:
            raise RuntimeError("Student visual hook did not capture features.")
        feat = self._captured_student_visual
        feat = _flatten_feature_to_tokens(feat).mean(dim=1)
        return self._pack_per_image_features(feat, image_counts)

    @torch.no_grad()
    def _extract_base_visual_preserve_features(
        self,
        inputs: Dict[str, torch.Tensor],
        image_counts: Sequence[int],
    ) -> torch.Tensor:
        vision_outputs = self.base_vision_tower(
            pixel_values=inputs["pixel_values"],
            pixel_position_ids=inputs.get("image_position_ids"),
            return_dict=True,
        )
        feat = getattr(vision_outputs, "pooler_output", None)
        if feat is None:
            feat = getattr(vision_outputs, "last_hidden_state", None)
        if feat is None:
            raise RuntimeError("Base vision tower output has neither pooler_output nor last_hidden_state.")
        feat = _flatten_feature_to_tokens(feat).mean(dim=1)
        return self._pack_per_image_features(feat, image_counts)

    def forward(self, images: Sequence[Union[Image.Image, Sequence[Image.Image]]], options_batch: Sequence[Sequence[str]],
                labels: Sequence[int], question_texts: Optional[Sequence[str]] = None,
                wan_captions: Optional[Sequence[str]] = None,
                wan_caption_embeddings: Optional[Sequence[Optional[torch.Tensor]]] = None) -> ForwardOutput:
        image_counts = self._image_counts(images)
        align_token_len = 2 if self.teacher_mode == "i2v_pair" else 1
        h0_vis = None
        if self.needs_base and self.preserve_mode == "vision_only":
            base_inputs = self._prepare_inputs(images, options_batch, question_texts)
            with torch.no_grad():
                h0_vis = self._extract_base_visual_preserve_features(base_inputs, image_counts).detach()

        self._captured_student_visual = None
        inputs = self._prepare_inputs(images, options_batch, question_texts)
        batch_size = len(images)
        zero = torch.tensor(0.0, device=self.device)

        if self.align_only:
            logits, h = self._extract_projected_image_tokens(
                self.model, inputs, batch_size, image_counts=image_counts, target_token_len=align_token_len
            )
            loss_task = zero
            option_logits = torch.zeros(batch_size, 6, device=self.device)
        elif self.task_only:
            outputs = self.model(
                **inputs, return_dict=True, output_hidden_states=False, use_cache=False,
            )
            logits = outputs.logits
            next_logits = logits[:, -1, :]
            letter_ids = self.option_token_ids.to(next_logits.device)
            option_logits = next_logits.index_select(dim=-1, index=letter_ids)
            target = torch.tensor(labels, dtype=torch.long, device=option_logits.device)
            loss_task = F.cross_entropy(option_logits, target)
            return ForwardOutput(
                loss=loss_task, loss_task=loss_task, loss_align=zero, loss_preserve=zero,
                logits=option_logits, pred_indices=option_logits.argmax(dim=-1),
            )
        else:
            logits, h = self._extract_projected_image_tokens(
                self.model, inputs, batch_size, image_counts=image_counts, target_token_len=align_token_len
            )
            next_logits = logits[:, -1, :]
            letter_ids = self.option_token_ids.to(next_logits.device)
            option_logits = next_logits.index_select(dim=-1, index=letter_ids)
            target = torch.tensor(labels, dtype=torch.long, device=option_logits.device)
            loss_task = F.cross_entropy(option_logits, target)

        # Alignment loss — teacher runs on teacher_device, move result to student_device
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

        if self.dual_timesteps:
            feats = []
            orig_t = self.lingbot_teacher.target_timestep
            for ts in self.dual_timesteps:
                self.lingbot_teacher.target_timestep = ts
                feat = self.lingbot_teacher(images, target_token_len=1,
                                            prompts=wan_prompts,
                                            prompt_embeddings=wan_embeds)
                feats.append(feat.to(device=self.device, dtype=torch.bfloat16))
            self.lingbot_teacher.target_timestep = orig_t
            teacher_feat = torch.cat(feats, dim=-1)  # [B, 1, 2*D]
        elif self.dual_blocks:
            self._dual_captured = [None] * len(self.dual_blocks)
            _ = self.lingbot_teacher(images, target_token_len=1,
                                     prompts=wan_prompts,
                                     prompt_embeddings=wan_embeds)
            feats = []
            for cap in self._dual_captured:
                f = _flatten_feature_to_tokens(cap)
                f = _pool_tokens_to_len(f, 1)
                feats.append(f.to(device=self.device, dtype=torch.bfloat16))
            teacher_feat = torch.cat(feats, dim=-1)  # [B, 1, N*D]
        else:
            teacher_feat = self.lingbot_teacher(images, target_token_len=1,
                                               prompts=wan_prompts,
                                               prompt_embeddings=wan_embeds)
            teacher_feat = teacher_feat.to(device=self.device, dtype=torch.bfloat16)
        z_m = self.student_align(h)
        z_g = self.teacher_align(teacher_feat)
        loss_align = _cosine_align(z_m, z_g.detach())

        # Preserve loss
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
