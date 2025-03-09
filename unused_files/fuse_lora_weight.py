from ltx_video_lora import *



pipe = initialize_pipeline(model_id="./data/fused-bk")


lora_path = "./data/49frames-first"
lora_rank = 128
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="default")
print("fuse lora")
pipe.fuse_lora(lora_scale=1.0, components=["transformer"])
# ----------- save
pipe.unload_lora_weights()
pipe.save_pretrained("./data/fused")