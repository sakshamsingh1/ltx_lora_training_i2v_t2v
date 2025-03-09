import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
import os, random
from dataset import PrecomputedDataset

model_id = "a-r-r-o-w/LTX-Video-0.9.1-diffusers"
COUNT = 100
LORA_WEIGHT = 0.4
# model_id = "./data/fused"

pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# ---------
lora_path = lora_path = "/home/eisneim/www/ml/_video_gen/ltx_training/data/mixed-bk/checkpoint-20000"
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="orbit")
pipe.set_adapters("orbit", LORA_WEIGHT)
# ----------

pipe.to("cuda")

dataset = PrecomputedDataset(data_dir="/media/eisneim/4T/ltx_data_121")
# start_idx = random.randint(0, len(dataset) - COUNT - 1)
start_idx = 266

negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
# prefix = "freeze time, camera orbit left, " 

for ii in range(start_idx, start_idx + COUNT):
    _, _, _, cap, info = dataset[ii]
    prompt = cap
    # prompt  = prefix + prompt
    print(prompt)
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=576,
        num_frames=81,
        # width=512,
        # height=288,
        # num_frames=121,
        num_inference_steps=50,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
    ).frames[0]
    export_to_video(video, f"data/outputs/1-19/lora_{LORA_WEIGHT}_{ii}.mp4", fps=24)

print("done!")