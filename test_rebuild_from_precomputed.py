import os, random, math
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2
import torch
from torchvision import transforms
from PIL import Image
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

from ltx_video_lora import *
device = "cuda"
dtype = torch.bfloat16
# ------------------- 

vae = load_latent_models()["vae"].to(device, dtype=dtype)

def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents

def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0,
    reverse=False,
) -> torch.Tensor:
    # Normalize latents across the channel dimension [B, C, F, H, W]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    if not reverse:
        latents = (latents - latents_mean) * scaling_factor / latents_std
    else:
        latents = latents * latents_std / scaling_factor + latents_mean
    return latents

file = "/media/eisneim/4T/ltx_data/game_p4_49x1024x576/wd_1735956772_0.pt"
data = torch.load(file)
ll = data["latents"][0].unsqueeze(0)
print(ll.shape)

lt = _unpack_latents(ll.to(device, dtype=dtype), 7, 576//32,  1024//32)
# denormolize
lt = _normalize_latents(lt, vae.latents_mean, vae.latents_std, reverse=True)

print(lt.shape)

timestep = torch.tensor([0.05], device=device, dtype=dtype)

with torch.no_grad():
    video =  vae.decode(lt, timestep, return_dict=False)[0]
pcc = VideoProcessor(vae_scale_factor=32)
vv = pcc.postprocess_video(video)[0]
export_to_video(vv, "data/test_gamep4_1024x576-1-t0.05.mp4", fps=24)