import torch
import numpy as np
import librosa

from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

from diffusers import AutoencoderKLLTXVideo
MODEL_ID = "a-r-r-o-w/LTX-Video-0.9.1-diffusers"


SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 122 #256
N_MELS = 128  # for example

MIN_DB = -80.0  # global minimum decibel
MAX_DB = 0.0    # global maximum decibel


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

def audio_to_spectrogram(audio_path, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db_clipped = np.clip(mel_db, MIN_DB, MAX_DB)
    mel_norm = (mel_db_clipped - MIN_DB) / (MAX_DB - MIN_DB)
    mel_255 = (mel_norm * 255.0).astype(np.uint8)

    mel_255 = mel_255[..., np.newaxis]              # shape (N_MELS, Time, 1)
    mel_255 = np.repeat(mel_255, 3, axis=2)         # shape (N_MELS, Time, 3)
    # add a frame dimension
    mel_255 = mel_255[np.newaxis, ...]               # shape (1, N_MELS, Time, 3)
    return mel_255

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

def spectrogram_to_audio(mel_255, sample_rate=SAMPLE_RATE):
    mel_min, mel_max = MIN_DB, MAX_DB

    if mel_255.shape[-1] == 3:
        mel_255 = mel_255[..., 0]

    mel_norm = mel_255.astype(np.float32) / 255.0
    mel_db = mel_norm * (mel_max - mel_min) + mel_min  # exact dB range
    mel_power = librosa.db_to_power(mel_db, ref=1.0)

    melfb = librosa.filters.mel(sr=sample_rate, n_fft=N_FFT, n_mels=mel_power.shape[0])
    inv_linear = np.dot(np.linalg.pinv(melfb), mel_power)

    # inv_linear = inv_linear.astype(np.float32)
    audio = librosa.griffinlim(inv_linear, hop_length=HOP_LENGTH, n_iter=32)
    return audio

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

def to_tensor(spectrogram_data, device="cuda", dtype=torch.bfloat16):
    input_ = (spectrogram_data / 255.0) * 2.0 - 1.0
    tensor = torch.from_numpy(input_).permute(0, 3, 1, 2).unsqueeze(0)  # (1,3,H,W)
    tensor = tensor.to(device, dtype=dtype)
    return tensor

def export_to_wav(audio, save_path, sr=SAMPLE_RATE):
    # Save to .wav using librosa
    import soundfile as sf
    sf.write(save_path, audio, sr)

class VAEAudioAnalyse:
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.vae = load_latent_models()["vae"].to(self.device, dtype=self.dtype)

    def audio_to_latent(self, audio_path):
        mel_255 = audio_to_spectrogram(audio_path)
        frame_t = to_tensor(mel_255, self.device, self.dtype)

        with torch.no_grad():
            data = prepare_latents(
                vae=self.vae,
                image_or_video=frame_t,  # shape (1, 3, H, W)
                device=self.device,
                dtype=self.dtype,
            )

        return data["latents"].cpu().to(self.dtype)

    def latent_to_audio(self, latents, save_path="output.wav"):
        """
        1) Decode latents -> spectrogram
        2) Convert spectrogram -> audio
        3) Save to disk
        """
        latents = latents.to(self.device, self.dtype)
        
        latents = _unpack_latents(latents, 1, N_MELS//32, 1312//32)
        latents = _normalize_latents(latents, self.vae.latents_mean, self.vae.latents_std, reverse=True)

        timestep = torch.tensor([0.05], device=self.device, dtype=self.dtype)
        with torch.no_grad():
            # shape: (B=1, 3, H, W)
            spectrogram_decoded = self.vae.decode(latents, timestep, return_dict=False)[0]

        # print(spectrogram_decoded.shape)

        # Convert [-1..1] float back to [0..255] uint8
        spectrogram_decoded = spectrogram_decoded[:,:,0][0]  # drop batch dim => (3, W, H)

        spectrogram_decoded = spectrogram_decoded.permute(1,2,0).cpu().float().numpy()  # => (H, W, 3)
        spectrogram_decoded = (spectrogram_decoded + 1.0) * 127.5
        spectrogram_decoded = np.clip(spectrogram_decoded, 0, 255) #.astype(np.uint8)

        # Now invert that spectrogram to audio
        audio = spectrogram_to_audio(spectrogram_decoded)

        export_to_wav(audio, save_path)
        # return spectrogram_decoded
