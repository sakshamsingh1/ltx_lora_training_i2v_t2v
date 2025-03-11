import torch
import json
from huggingface_hub import snapshot_download
import torch_tools as torch_tools
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL

class AudioLDM_VAE():
    def __init__(self, device="cuda:0"):
        vae, stft = self.get_vae_stft(device=device)
        self.vae = vae
        self.stft = stft

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

    def audio_to_latent(self, aud_paths, target_length=1024):
        mel, _, waveform = torch_tools.wav_to_fbank(list(aud_paths), target_length, self.stft)
        mel = mel.unsqueeze(1)

        true_latent = self.vae.get_first_stage_encoding(self.vae.encode_first_stage(mel))
        return true_latent
    
    def latent_to_audio(self, latent):
        mel = self.vae.decode_first_stage(latent)
        wave = self.vae.decode_to_waveform(mel)[0]
        return wave