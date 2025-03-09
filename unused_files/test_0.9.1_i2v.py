import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

model_id = "./data/fused"
# pipe = LTXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = LTXImageToVideoPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)

lora_path = "/home/eisneim/www/ml/_video_gen/ltx_training/data/mixed/checkpoint-13000"
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="default")
pipe.to("cuda")


image = load_image("data/images/8.png")

prefix = "freeze time, camera orbit left, " 
# prompt = "a person with short black hair and light skin is standing in the middle of street, cars in the background"
prompt = "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background. Flames engulf the structure, with smoke billowing into the air. Firefighters in protective gear rush to the scene, a fire truck labeled '38' visible behind them. The girl's neutral expression contrasts sharply with the chaos of the fire, creating a poignant and emotionally charged scene."
prompt  = prefix + prompt

negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_frames=161,
    num_inference_steps=50,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
).frames[0]
export_to_video(video, "data/i2v_15k.mp4", fps=24)