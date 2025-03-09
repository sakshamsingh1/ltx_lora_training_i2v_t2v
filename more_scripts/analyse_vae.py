import torch
import numpy as np
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

from diffusers import AutoencoderKLLTXVideo
MODEL_ID = "a-r-r-o-w/LTX-Video-0.9.1-diffusers"

#### helper functions
def load_latent_models(
    model_id = MODEL_ID,
    vae_dtype = torch.bfloat16,
    revision = None,
    cache_dir = None):
    vae = AutoencoderKLLTXVideo.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"vae": vae}

def _pack_latents(latents, patch_size = 1, patch_size_t = 1):
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents

def _unpack_latents(
        latents, num_frames, height, width, patch_size = 1, patch_size_t = 1
    ):
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents

def _normalize_latents(
    latents, latents_mean, latents_std, scaling_factor = 1.0, reverse=False,
):
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    if not reverse:
        latents = (latents - latents_mean) * scaling_factor / latents_std
    else:
        latents = latents * latents_std / scaling_factor + latents_mean
    return latents

def prepare_latents(
    vae,
    image_or_video,
    patch_size = 1,
    patch_size_t = 1,
    device = None,
    dtype = None,
    generator = None,
    precompute = False,
):
    device = device or vae.device

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]
    if not precompute:
        latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
        latents = latents.to(dtype=dtype)
        _, _, num_frames, height, width = latents.shape
        latents = _normalize_latents(latents, vae.latents_mean, vae.latents_std)
        latents = _pack_latents(latents, patch_size, patch_size_t)
        return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}
    else:
        if vae.use_slicing and image_or_video.shape[0] > 1:
            encoded_slices = [vae._encode(x_slice) for x_slice in image_or_video.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = vae._encode(image_or_video)
        _, _, num_frames, height, width = h.shape

        return {
            "latents": h,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "latents_mean": vae.latents_mean,
            "latents_std": vae.latents_std,
        }

def get_frame(image_path, w = None, h = None):
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    if w is not None and h is not None:
        img = img.resize((w, h))  
    img_array = np.array(img, dtype=np.uint8)
    return img_array[np.newaxis, ...]  # (1, h, w, 3)

def to_tensor(data, device="cuda", dtype=torch.bfloat16):
    input = (data / 255) * 2.0 - 1.0
    return torch.from_numpy(input).permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype=dtype)    

class VAE_analyse:
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.vae = load_latent_models()["vae"].to(self.device, dtype=self.dtype)

    def img_to_latent(self, frame_path, w = 768, h = 512):
        frame = get_frame(frame_path, w, h)
        frame_t = to_tensor(frame, self.device, self.dtype)

        with torch.no_grad():
            data = prepare_latents(
                    vae=self.vae,
                    image_or_video=frame_t,
                    device=self.device,
                    dtype=self.dtype,
                )

        return data["latents"].cpu().to(self.dtype)

    def latent_to_img(self, img_latent, save_path, w = 768, h = 512):
        num_frames = 1
        img_latent = img_latent.to(self.device, self.dtype)
        img_latent = _unpack_latents(img_latent, (num_frames+7)//8, h//32, w//32)
        # denormolize
        img_latent = _normalize_latents(img_latent, self.vae.latents_mean, self.vae.latents_std, reverse=True)

        timestep = torch.tensor([0.05], device=self.device, dtype=self.dtype)
        with torch.no_grad():
            image =  self.vae.decode(img_latent, timestep, return_dict=False)[0]

        pcc = VideoProcessor(vae_scale_factor=32)
        vv = pcc.postprocess_video(image)[0]

        export_to_video(vv, save_path, fps=24)