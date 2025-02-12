import os, random, math, time
import numpy as np
import cv2
import torch

from ltx_video_lora import *


"""
CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python text_embedding_test.py 


"""
if __name__ == "__main__":

    prompt_prefix = "freeze time, camera orbit left,"
    device = "cuda"
    dtype = torch.bfloat16
 
    cond_models = load_condition_models()
    tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
    text_encoder = text_encoder.to(device, dtype=dtype)
            
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


    prompt_embeds, prompt_attention_mask = to_embedding(prompt_prefix)
    print("prompt_embeds", prompt_embeds.shape, prompt_embeds)
    print("prompt_attention_mask", prompt_attention_mask.shape, prompt_attention_mask)
    torch.save(dict(prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_attention_mask), "data/default_prompt_embedding.pth")


