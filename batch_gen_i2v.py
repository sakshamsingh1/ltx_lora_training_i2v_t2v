
import os, random

import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


LORA_WEIGHT = 0.3
guidance = 3
# lora_path = "/home/eisneim/www/ml/_video_gen/ltx_training/data/mixed-2-8/checkpoint-13000"
lora_path = "/home/eisneim/www/ml/_video_gen/ltx_training/data_pissa/mixed/checkpoint-19000"
prefix = "freeze time, camera orbit left, " 
# negative_prompt = "worst quality, static, noisy, inconsistent motion, blurry, jittery, distorted"
negative_prompt = "worst quality, static no camera movement, noisy, inconsistent motion, blurry, jittery, distorted"

images_dir = "data/images/dimx_part1/"
output_dir = "data_pissa/outputs/i2v_2-8-2"

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

pipe = LTXImageToVideoPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)


pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="orbit")
pipe.set_adapters("orbit", LORA_WEIGHT)

pipe.to("cuda")

for idx in range(len(pairs)):
    file, prompt = pairs[idx]
    print(f"[{idx}/{len(pairs)}] {file}")
    prompt = prefix + prompt
    print(prompt)
    fname = os.path.basename(file).replace(".jpg", "") + f"_lora-{LORA_WEIGHT}_g-{guidance}-2.mp4"
    dest = os.path.join(output_dir, fname)
    
    width = 704
    height = 480
    image = load_image(file).resize((width, height))

    video = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=121,
        num_inference_steps=50,
        decode_timestep=0.05,
        decode_noise_scale=0.025,
        guidance_scale=guidance,
    ).frames[0]
    export_to_video(video, dest, fps=24)

print("done!")