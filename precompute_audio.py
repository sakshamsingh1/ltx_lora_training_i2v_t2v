import os
from tqdm import tqdm
from pathlib import Path
import json
import torch
import pandas as pd

# os.environ['all_proxy']=''
# os.environ['all_proxy']=''

from icecream import ic
from constants import DEFAULT_HEIGHT_BUCKETS, DEFAULT_WIDTH_BUCKETS, DEFAULT_FRAME_BUCKETS
from audio_helpers.aud_latent_utils import load_wav, get_mel_spectrogram_from_audio, normalize_spectrogram, pad_spec, normalize, to_tensor
# if there is a memery leak in the code, we'll shut it down manually
import psutil

ic.disable()

memusage = psutil.virtual_memory()[2]
assert memusage < 85, "Impending memory leak, memory needs to be cleared before running."

class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        audio_dir, # list or string path
        aud_list_file: str,
        cache_dir: str,
        spec_time_bins: int = 512, # ~5 sec
        mel_bins: int = 256,
        prompt_prefix = "freeze time, camera orbit left,",
        device = "cuda",
        dtype = torch.bfloat16,
    ):
        super().__init__()
        assert spec_time_bins in DEFAULT_WIDTH_BUCKETS, f"width only supported in: {DEFAULT_WIDTH_BUCKETS}"
        assert mel_bins in DEFAULT_HEIGHT_BUCKETS, f"height only supported in: {DEFAULT_HEIGHT_BUCKETS}"
        
        self.spec_time_bins = spec_time_bins
        self.mel_bins = mel_bins
        self.audio_dir = audio_dir
        
        self.cache_dir = Path(f"{cache_dir}_{self.mel_bins}x{self.spec_time_bins}")
        os.makedirs(self.cache_dir, exist_ok=True)
        print("cache_dir:", self.cache_dir)
        
        self.device = device
        self.dtype = dtype
        self.prompt_prefix = prompt_prefix
        
        self.audios = []
        self.data = []

        audio_file = aud_list_file
        with open(audio_file, "r") as f:
            aud_ids = [os.path.basename(line.strip()).replace('.mp4','.wav') for line in f.readlines()]
            self.audios = [os.path.join(audio_dir, aud_id) for aud_id in aud_ids]
        
        #################### Quick test ####################
        self.audios = self.audios[:10]
        #################### Quick test ####################

        print(f"{type(self).__name__} found {len(self.audios)} videos ")
    
    def get_norm_spec(self, audio_path):
        audio, sr = load_wav(audio_path)
        audio, spec = get_mel_spectrogram_from_audio(audio)
        norm_spec = normalize_spectrogram(spec)
        norm_spec = pad_spec(norm_spec, self.spec_time_bins)
        norm_spec = normalize(norm_spec)
        norm_spec = to_tensor(norm_spec, self.device, self.dtype)
        return norm_spec

    def cache_frames(self, to_latent, to_caption, to_embedding):
        print(f"building caches, video count: {len(self.audios)}")

        for ii, aud_path in enumerate(tqdm(self.audios)):
            dest = os.path.join(self.cache_dir, os.path.basename(aud_path).rsplit(".", 1)[0] + ".pt")
            ic(dest)
            if os.path.exists(dest):
                # print("skip:", dest)
                continue
            try:
                norm_spec = self.get_norm_spec(aud_path)
            except Exception as e:
                print("error file:", aud_path, e)
                continue
            # remove first frame for frame blending motion blured video

            # divid into parts
            iters = 1 # let's fix it to 1 for now
            latents = []
            embedds = []
            masks = []
            captions = []
            infos = []

            for idx in range(iters):
                caption = self.prompt_prefix + " " + to_caption(aud_path)
                ic(caption)
                embedding, mask = to_embedding(caption.replace("  ", " "))
                ic(embedding.shape, mask.shape)

                ic(norm_spec)
                latent, num_frames, height, width = to_latent(norm_spec)
                assert latent.ndim == 3, "patched latent should have 3 dims"
                ic(latent.shape, latent)
                # make sure is not nan
                
                captions.append(caption)
                embedds.append(embedding)
                masks.append(mask)
                latents.append(latent)
                infos.append(dict(num_frames=num_frames, height=height, width=width))

            latents = torch.cat(latents, dim=0)
            embedds = torch.cat(embedds, dim=0)
            masks = torch.cat(masks, dim=0)
            
            # print(latent.shape, latent_lr.shape)
            # np.savez(dest, hr=latent, lr=latent_lr)
            torch.save(dict(latents=latents, 
              embedds=embedds, 
              masks=masks, 
              captions=captions, 
              meta_info=infos), dest)
            self.data.append(dest)

            memusage = psutil.virtual_memory()[2]
            assert memusage < 80, "即将内存泄漏，强行关闭，请重新启动进程"
            
        print(f">> cached {len(self.data)} videos")
    
    def __getitem__(self, idx):
        return self.audios[idx]

class Qwen_Captions():
    def __init__(self):
        self.cog_caption_path = '/mnt/ssd0/saksham/i2av/AVSync15/aud_caption.json'
        self.caption_map = json.load(open(self.cog_caption_path, "r"))
        self.vid_label_map = self.get_label_map()
    
    def get_label_map(self):
        meta_path='/home/sxk230060/ltx_lora_training_i2v_t2v/preprocess/asva_metadata.csv'
        df = pd.read_csv(meta_path)
        return dict(zip(df["vid"], df["label"]))

    def get_caption(self, aud_path):
        vid = os.path.basename(aud_path).rsplit(".", 1)[0]
        label = self.vid_label_map[vid]
        label = label.replace("__", " ").replace("_", " ")
        text_caption = label + ", " + self.caption_map[vid]
        return text_caption

if __name__ == "__main__":
    import argparse
    from yaml import load, Loader
    from tqdm import tqdm

    from ltx_video_lora import *
    
    aud_base_dir = "/mnt/ssd0/saksham/i2av/AVSync15/audios_together"
    aud_list_file = '/home/sxk230060/TI2AV/data/AVSync15/train.txt'
    cache_dir = "/mnt/ssd0/saksham/i2av/ltx_lora_training_i2v_t2v/cache_audio"
    
    config_file = "./configs/ltx_audio.yaml"
    device = "cuda"
    dtype = torch.bfloat16
    prompt_prefix = "sounding object, "
    # ------------------- 

    config_dict = load(open(config_file, "r"), Loader=Loader)
    args = argparse.Namespace(**config_dict)

    # ----------- prepare models -------------
    dataset = AudioDataset(
        audio_dir=aud_base_dir,
        aud_list_file=aud_list_file,
        cache_dir=cache_dir,
        mel_bins=args.mel_bins,
        spec_time_bins=args.spec_time_bins,
        prompt_prefix=prompt_prefix,
        device = device,
        dtype = dtype
    )

    captioner = Qwen_Captions()
    cond_models = load_condition_models()
    tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
    text_encoder = text_encoder.to(device, dtype=dtype)
    vae = load_latent_models()["vae"].to(device, dtype=dtype)

    def to_latent(norm_spec):
        ic(norm_spec.shape)
        assert norm_spec.size(2) == 3, f"frames should be in shape: (b, f, c, h, w) provided: {norm_spec.shape}"
        with torch.no_grad():
            data = prepare_latents(
                    vae=vae,
                    image_or_video=norm_spec,
                    device=device,
                    dtype=dtype,
                )
        return data["latents"].cpu().to(dtype), data["num_frames"], data["height"], data["width"]
            
    def to_embedding(caption):
        with torch.no_grad():
            text_conditions = prepare_conditions(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=caption,
            )
            prompt_embeds = text_conditions["prompt_embeds"].to("cpu", dtype=dtype)
            prompt_attention_mask = text_conditions["prompt_attention_mask"].to("cpu", dtype=dtype)
        return prompt_embeds, prompt_attention_mask
    
    dataset.cache_frames(to_latent, captioner.get_caption, to_embedding)
    print("done!")