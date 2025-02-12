import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

model_id = "a-r-r-o-w/LTX-Video-0.9.1-diffusers"
# model_id = "./data/fused"

pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# ---------
lora_path = lora_path = "/home/eisneim/www/ml/_video_gen/ltx_training/data/49frames"
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="orbit")
pipe.set_adapters("orbit", 0.98)
# ----------

pipe.to("cuda")

prefix = "freeze time, camera orbit left, " 
prompt = "a person with short black hair and light skin is standing in the middle of street, cars in the background"
prompt  = prefix + prompt
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

for ii in range(3):
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
    export_to_video(video, f"data/lora_15k_{ii}.mp4", fps=24)