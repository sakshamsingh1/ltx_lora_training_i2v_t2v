import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torch

import json
from huggingface_hub import snapshot_download
# import audioldm_vae.torch_tools as torch_tools
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video_lora import prepare_conditions, _normalize_latents, _pack_latents, load_latent_models

class AudioLDM_dataset(Dataset):
    def __init__(self, device="cuda:0", split="train", prefix="sounding object, ", dtype=torch.bfloat16):
        vae, stft = self.get_vae_stft(device=device)
        self.vae = vae
        self.stft = stft
        self.aud_base_dir = "/mnt/sda1/saksham/TI2AV/AVSync15/audios_together"
        aud_caption_path = '/mnt/sda1/saksham/TI2AV/AVSync15/aud_caption.json'
        self.aud_caption = json.load(open(aud_caption_path, "r"))
        aud_meta_path = "/mnt/sda1/saksham/TI2AV/AVSync15/metadata.csv"
        df = pd.read_csv(aud_meta_path)
        self.df = df[df['split']==split].reset_index(drop=True)
        self.prefix = prefix

        self.device = device
        self.dtype = dtype
        tokenizer, text_encoder = self.load_condition_models()
        self.tokenizer = tokenizer; self.text_encoder = text_encoder

        self.ltx_vae = load_latent_models()["vae"].to(self.device, dtype=self.dtype)
        self.patch_size=1; self.patch_size_t=1

    def load_condition_models(self,
        model_id = "a-r-r-o-w/LTX-Video-0.9.1-diffusers",
        text_encoder_dtype = torch.bfloat16,
        revision = None,
        cache_dir = None,
        ):
        
        tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision, cache_dir=cache_dir)
        text_encoder = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=text_encoder_dtype, revision=revision, cache_dir=cache_dir
        )

        return tokenizer, text_encoder

    def get_vae_stft(self, name="declare-lab/tango", device="cuda:0"):
        path = snapshot_download(repo_id=name)
        vae_config = json.load(open("{}/vae_config.json".format(path)))
        stft_config = json.load(open("{}/stft_config.json".format(path)))
        vae = AutoencoderKL(**vae_config).to(device)
        stft = TacotronSTFT(**stft_config).to(device)
        vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path), map_location=device)
        stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path), map_location=device)
        vae.load_state_dict(vae_weights)
        stft.load_state_dict(stft_weights)
        vae.eval()
        stft.eval()
        return vae, stft
    
    def audio_to_latent(self, aud_paths, target_length=512):
        mel, _, waveform = self.wav_to_fbank(list(aud_paths), target_length, self.stft)
        mel = mel.unsqueeze(1)
        true_latent = self.vae.get_first_stage_encoding(self.vae.encode_first_stage(mel))
        return true_latent

    def to_embedding(self, caption):
        with torch.no_grad():
            text_conditions = prepare_conditions(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                prompt=caption,
            )
            prompt_embeds = text_conditions["prompt_embeds"].to("cpu", dtype=self.dtype)
            prompt_attention_mask = text_conditions["prompt_attention_mask"].to("cpu", dtype=self.dtype)
        return prompt_embeds, prompt_attention_mask

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = row["vid"]; label = row["label"]
        aud_path = os.path.join(self.aud_base_dir, vid + ".wav")
        caption = self.prefix + label + ", " + self.aud_caption[vid]
        latent = self.audio_to_latent([aud_path]) # [B,C,W,H] = [1, 8, 128, 16]

        latent = latent.unsqueeze(1).permute(1, 2, 0, 4, 3) # [B,C,W,H] -> [F, B, C, W, H] -> [B, C, F, H, W] = [1, 8, 1, 16, 128]
        latent = latent.repeat(1, 16, 1, 1, 1) # [B, C, F, H, W] = [1, 8*16, 1, 16, 128]

        _, _, num_frames, height, width = latent.shape
        latent = _normalize_latents(latent, self.ltx_vae.latents_mean, self.ltx_vae.latents_std)
        latent = _pack_latents(latent, self.patch_size, self.patch_size_t)

        embeds, masks = self.to_embedding(caption)
        
        # todo repeat the latent vector
        return latent, embeds, masks
    
    ######################################################### torch_tools.py #########################################################

    def normalize_wav(self, waveform):
        waveform = waveform - torch.mean(waveform)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        return waveform * 0.5

    def pad_wav(self, waveform, segment_length):
        waveform_length = len(waveform)
        
        if segment_length is None or waveform_length == segment_length:
            return waveform
        elif waveform_length > segment_length:
            return waveform[:segment_length]
        else:
            pad_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
            waveform = torch.cat([waveform, pad_wav])
            return waveform
        
    def _pad_spec(self, fbank, target_length=1024):
        batch, n_frames, channels = fbank.shape
        p = target_length - n_frames
        if p > 0:
            pad = torch.zeros(batch, p, channels).to(fbank.device)
            fbank = torch.cat([fbank, pad], 1)
        elif p < 0:
            fbank = fbank[:, :target_length, :]

        if channels % 2 != 0:
            fbank = fbank[:, :, :-1]

        return fbank

    def read_wav_file(self, filename, segment_length):
        waveform, sr = torchaudio.load(filename)  # Faster!!!
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)[0]
        try:
            waveform = self.normalize_wav(waveform)
        except:
            print ("Exception normalizing:", filename)
            waveform = torch.ones(160000)
        waveform = self.pad_wav(waveform, segment_length).unsqueeze(0)
        waveform = waveform / torch.max(torch.abs(waveform))
        waveform = 0.5 * waveform
        return waveform

    def get_mel_from_wav(self, audio, _stft):
        audio = torch.nan_to_num(torch.clip(audio, -1, 1))
        audio = torch.autograd.Variable(audio, requires_grad=False)
        audio = audio.to(self.device)
        melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
        return melspec, log_magnitudes_stft, energy

    def wav_to_fbank(self, paths, target_length=1024, fn_STFT=None):
        assert fn_STFT is not None

        waveform = torch.cat([self.read_wav_file(path, target_length * 160) for path in paths], 0)  # hop size is 160

        fbank, log_magnitudes_stft, energy = self.get_mel_from_wav(waveform, fn_STFT)
        fbank = fbank.transpose(1, 2)
        log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

        fbank, log_magnitudes_stft = self._pad_spec(fbank, target_length), self._pad_spec(
            log_magnitudes_stft, target_length
        )

        return fbank, log_magnitudes_stft, waveform
