
import os, random

import torch
# from diffusers import LTXImageToVideoPipeline
from pipeline_stg_ltx_image2video import LTXImageToVideoSTGPipeline
from diffusers.utils import export_to_video, load_image

LORA_WEIGHT = 0.8
# width = 1024
# height = 576
# width = 736
# height = 416
width = 704
height = 480
num_frames=121
lora_path = "/home/eisneim/www/ml/video_gen/ltx_training/data/ckpt_27k"
prefix = "freeze time, camera orbit left, " 
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

images_dir = "data/images/dimx_part1/"
# images_dir = "data/portrait_img/"
output_dir = "data/outputs/i2v_2-11"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pairs = []
for ii in os.listdir(images_dir):
    if ii.endswith(".jpg") and ii[0] != ".":
        file = os.path.join(images_dir, ii)
        prompt = open(os.path.join(images_dir, ii.rsplit(".", 1)[0] + ".txt"), "r").read()
        prompt = prompt.replace("The image", "The video")
        pairs.append((file, prompt))

print("loaded images", len(pairs))
pairs.sort(key=lambda tt: os.path.basename(tt[0]))

pipe = LTXImageToVideoSTGPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16, local_files_only=True)


pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="orbit")
pipe.set_adapters("orbit", LORA_WEIGHT)

pipe.to("cuda")

stg_mode = "STG-A" # STG-A, STG-R
stg_applied_layers_idx = [19] # 0~27
stg_scale = 1.0 # 0.0 for CFG
do_rescaling = True # Default (False)


for idx in range(len(pairs)):
    file, prompt = pairs[idx]
    print(f"[{idx}/{len(pairs)}] {file}")
    prompt = prefix + prompt
    print(prompt)
    basename = os.path.basename(file).replace(".jpg", "")
    fname = f"{basename}_27k_{LORA_WEIGHT}_stglayer{stg_applied_layers_idx}.mp4"
    dest = os.path.join(output_dir, fname)
    
    image = load_image(file).resize((width, height))

    video = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=50,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        generator=None,
        stg_mode=stg_mode,
        stg_applied_layers_idx=stg_applied_layers_idx,
        stg_scale=stg_scale,
        do_rescaling=do_rescaling
    ).frames[0]
    export_to_video(video, dest, fps=24)

print("done!")