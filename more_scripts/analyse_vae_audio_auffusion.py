
import os
from huggingface_hub import snapshot_download
import torch.nn.functional as F
from scipy.io.wavfile import read
import random
from auffusion_vocoder import Generator
from diffusers.video_processor import VideoProcessor
from scipy.io.wavfile import write

import torch
import numpy as np
import librosa

from librosa.filters import mel as librosa_mel_fn

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


#############################################################################################################
MAX_WAV_VALUE = 32768.0
mel_basis = {}
hann_window = {}

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    return spec

def get_mel_spectrogram_from_audio(audio, device="cuda"):
    audio = audio / MAX_WAV_VALUE
    audio = librosa.util.normalize(audio) * 0.95
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)    
    waveform = audio.to(device)
    spec = mel_spectrogram(waveform, n_fft=2048, num_mels=256, sampling_rate=16000, hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False)
    return audio, spec

def normalize_spectrogram(
    spectrogram,
    max_value = 200, 
    min_value = 1e-5, 
    power = 1., 
    inverse = False
    ):   
    # Rescale to 0-1
    max_value = np.log(max_value) # 5.298317366548036
    min_value = np.log(min_value) # -11.512925464970229
    assert spectrogram.max() <= max_value and spectrogram.min() >= min_value
    data = (spectrogram - min_value) / (max_value - min_value)
    if inverse:
        data = 1 - data
    data = torch.pow(data, power)  
    data = data.repeat(3, 1, 1)
    data = torch.flip(data, [1])
    return data

def denormalize_spectrogram(
    data: torch.Tensor,
    max_value = 200, 
    min_value = 1e-5, 
    power = 1, 
    inverse = False,
    ):
    max_value = np.log(max_value)
    min_value = np.log(min_value)
    data = torch.flip(data, [1])
    assert len(data.shape) == 3, "Expected 3 dimensions, got {}".format(len(data.shape))
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
    assert data.shape[0] == 3, "Expected 3 channels, got {}".format(data.shape[0])
    data = data[0]
    data = torch.pow(data, 1 / power)
    if inverse:
        data = 1 - data
    spectrogram = data * (max_value - min_value) + min_value
    return spectrogram

def pad_spec(spec, spec_length, pad_value=0, random_crop=True): # spec: [3, mel_dim, spec_len]
    assert spec_length % 8 == 0, "spec_length must be divisible by 8"
    if spec.shape[-1] < spec_length:
        # pad spec to spec_length
        spec = F.pad(spec, (0, spec_length - spec.shape[-1]), value=pad_value)
    else:
        # random crop
        if random_crop:
            start = random.randint(0, spec.shape[-1] - spec_length)
            spec = spec[:, :, start:start+spec_length]
        else:
            spec = spec[:, :, :spec_length]
    return spec

def normalize(images):
    if images.min() >= 0:
        return 2.0 * images - 1.0
    else:
        return images

def to_tensor(spec_data, device="cuda", dtype=torch.bfloat16):
    tensor = spec_data.unsqueeze(0).unsqueeze(0)
    # tensor = spec_data.permute(0, 3, 1, 2).unsqueeze(0)  # (1,3,H,W)
    tensor = tensor.to(device, dtype=dtype)
    return tensor

def get_vocoder(device, dtype):
    pretrained_model_name_or_path = "auffusion/auffusion-full-no-adapter"
    if not os.path.isdir(pretrained_model_name_or_path):
        pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)
    vocoder = Generator.from_pretrained(pretrained_model_name_or_path, subfolder="vocoder")
    vocoder = vocoder.to(device=device, dtype=dtype)  
    return vocoder 

class VAEAudioAnalyse:
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16
        self.vae = load_latent_models()["vae"].to(self.device, dtype=self.dtype)
        self.vocoder = get_vocoder(self.device, self.dtype)

    def audio_to_latent(self, audio_path):
        audio, sr = load_wav(audio_path)
        audio, spec = get_mel_spectrogram_from_audio(audio)
        norm_spec = normalize_spectrogram(spec)
        norm_spec = pad_spec(norm_spec, 1024)
        norm_spec = normalize(norm_spec)
        frame_t = to_tensor(norm_spec, self.device, self.dtype)

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
        
        latents = _unpack_latents(latents, 1, 256//32, 1024//32)
        latents = _normalize_latents(latents, self.vae.latents_mean, self.vae.latents_std, reverse=True)

        timestep = torch.tensor([0.05], device=self.device, dtype=self.dtype)
        with torch.no_grad():
            # shape: (B=1, 3, H, W)
            spectrogram_decoded = self.vae.decode(latents, timestep, return_dict=False)[0]

        # spectrogram_decoded = spectrogram_decoded[:,:,0][0]
        pcc = VideoProcessor(vae_scale_factor=32)
        vv = pcc.postprocess_video(spectrogram_decoded)[0]

        output_spec = torch.from_numpy(vv)[0].permute(2, 0, 1)
        output_spec = output_spec.to(device=self.device, dtype=self.dtype)
        denorm_spec = denormalize_spectrogram(output_spec)
        # import pdb; pdb.set_trace()
        
        spec_audio = self.vocoder.inference(denorm_spec)
        write(save_path, 16000, spec_audio[0])
        # export_to_wav(spec_audio, save_path)
