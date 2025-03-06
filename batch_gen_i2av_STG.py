import os, random
import json
import pandas as pd
from moviepy.editor import VideoFileClip, AudioFileClip

import torch
from diffusers import LTXPipeline
from pipeline_stg_ltx_image2video import LTXImageToVideoSTGPipeline
from more_scripts.analyse_vae_audio_auffusion import VAEAudioAnalyse
from diffusers.utils import export_to_video, load_image

def combine(video_path, audio_path, output_path, remove=True):
    video_clip = VideoFileClip(video_path)
    new_audio_clip = AudioFileClip(audio_path)
    video_with_new_audio = video_clip.set_audio(new_audio_clip)
    video_with_new_audio.write_videofile(output_path) # this plays on vscode
    # video_with_new_audio.write_videofile(output_path, codec='libx264', audio_codec='aac') # this plays on mac
    if remove:
        os.remove(video_path)
        os.remove(audio_path)

####################### initialising parameters #######################
model_id = "a-r-r-o-w/LTX-Video-0.9.1-diffusers"
images_dir = "/mnt/ssd0/saksham/i2av/AVSync15/first_frame"

LORA_WEIGHT = 0.8
prefix = "sounding object"
aud_caption_path = ''

output_base_dir = "/mnt/ssd0/saksham/i2av/ltx_lora_training_i2v_t2v/outputs/"

#I2V
width_v = 768
height_v = 512
num_frames_v = 121
vid_caption_path = '/mnt/ssd0/saksham/i2av/AVSync15/vid_caption_map.json'
vid_caption_map = json.load(open(vid_caption_path))

stg_mode = "STG-A" # STG-A, STG-R
stg_applied_layers_idx = [19] # 0~27
stg_scale = 1.0 # 0.0 for CFG
do_rescaling = True # Default (False)

negative_prompt_v = "worst quality, inconsistent motion, blurry, jittery, distorted"
USE_LORA_V = False

lora_dir_v = "/mnt/ssd0/saksham/i2av/ltx_lora_training_i2v_t2v/i2v_lora"
pipe_v = LTXImageToVideoSTGPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, local_files_only=False)
if USE_LORA_V:
    pipe_v.load_lora_weights(lora_dir_v, weight_name="pytorch_lora_weights.safetensors", adapter_name="ltx_lora")
    pipe_v.set_adapters("ltx_lora", LORA_WEIGHT)
    output_base_dir += "lora_v"
pipe_v.to("cuda")

#I2A
aud_caption_path = '/mnt/ssd0/saksham/i2av/AVSync15/aud_caption.json'
aud_caption_map = json.load(open(aud_caption_path))
USE_LORA_A = False
lora_dir_a = '/mnt/ssd0/saksham/i2av/ltx_lora_training_i2v_t2v/only_audio_1000'
pipe_a = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, local_files_only=True)
if USE_LORA_A:
    pipe_a.load_lora_weights(lora_dir_a, weight_name="pytorch_lora_weights.safetensors", adapter_name="ltx_lora")
    pipe_a.set_adapters("ltx_lora", LORA_WEIGHT)
    output_base_dir += "_lora_a"
pipe_a.to("cuda")
negative_prompt_a = ""

if (not USE_LORA_A) and (not USE_LORA_V):
    output_base_dir += "no_lora"

os.makedirs(output_base_dir, exist_ok=True)

meta_path = '/home/sxk230060/ltx_lora_training_i2v_t2v/preprocess/asva_metadata.csv'
df = pd.read_csv(meta_path)
df = df[df['split']=='test'].reset_index(drop=True)
df = df.groupby("label").first().reset_index()

for idx, row in df.iterrows():
    label = row['label'].replace("__", " ").replace("_", " ")
    
    #I2V
    caption = vid_caption_map[row['vid']]
    image_path = os.path.join(images_dir, row['vid'] + ".jpg")
    image = load_image(image_path).resize((width_v, height_v))
    prompt_v = f'{prefix}, {label}, {caption}'
    save_path_v = os.path.join(output_base_dir, f'{row["vid"]}_v.mp4')

    video = pipe_v(
        image=image,
        prompt=prompt_v,
        negative_prompt=negative_prompt_v,
        width=width_v,
        height=height_v,
        num_frames=num_frames_v,
        num_inference_steps=50,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        generator=None,
        stg_mode=stg_mode,
        stg_applied_layers_idx=stg_applied_layers_idx,
        stg_scale=stg_scale,
        do_rescaling=do_rescaling
    ).frames[0]
    export_to_video(video, save_path_v, fps=24)

    #I2A
    caption = aud_caption_map[row['vid']]
    prompt_a = f'{prefix}, {label}, {caption}'
    save_path_a = os.path.join(output_base_dir, f'{row["vid"]}_a.wav')

    video = pipe_a(
        prompt=prompt_a,
        negative_prompt=negative_prompt_a,
        width=512,
        height=256,
        num_frames=1,
        num_inference_steps=50,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        output_type='latent'
    ).frames

    vae_obj = VAEAudioAnalyse(spec_time_bins=512)
    vae_obj.latent_to_audio(video, save_path_a)

    #Combine
    combine(save_path_v, save_path_a, os.path.join(output_base_dir, f'av_{row["vid"]}.mp4'))