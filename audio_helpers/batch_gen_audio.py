import torch
from diffusers import LTXPipeline
import sys
import pandas as pd
import json
import os

# path='/home/sxk230060/TI2AV/misc/ltx_lora_training_i2v_t2v/more_scripts/'
# sys.path.append(path)
from more_scripts.analyse_vae_audio_auffusion import VAEAudioAnalyse

model_id = "a-r-r-o-w/LTX-Video-0.9.1-diffusers"
LORA_WEIGHT = 0.8

pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, local_files_only=True)

lora_path = "/mnt/ssd0/saksham/i2av/ltx_lora_training_i2v_t2v/only_audio_1000/checkpoint-14000"
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="ltx_lora")
pipe.set_adapters("ltx_lora", LORA_WEIGHT)
# ----------
pipe.to("cuda")

negative_prompt = ""
prefix = "sounding object, "

base_aud_dir = '/home/sxk230060/ltx_lora_training_i2v_t2v/more_scripts/temp_1000'
os.makedirs(base_aud_dir, exist_ok=True)
meta_path = '/home/sxk230060/ltx_lora_training_i2v_t2v/preprocess/asva_metadata.csv'
caption_path = '/mnt/ssd0/saksham/i2av/AVSync15/aud_caption.json'
vid_caption_map = json.load(open(caption_path))
df = pd.read_csv(meta_path)
df = df[df['split']=='test'].reset_index(drop=True)
df = df.groupby("label").first().reset_index()

for idx, row in df.iterrows():
    label = row['label'].replace("__", " ").replace("_", " ")
    caption = vid_caption_map[row['vid']]
    prompt = prefix + f'{label}, {caption}'

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=256,
        num_frames=1,
        num_inference_steps=50,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        output_type='latent'
    ).frames

    vae_obj = VAEAudioAnalyse(spec_time_bins=512)
    save_path = os.path.join(base_aud_dir, f'{row["label"]}.wav')
    vae_obj.latent_to_audio(video, save_path)