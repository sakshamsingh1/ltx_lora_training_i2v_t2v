import os
import torch
import json
from tqdm import tqdm
import cv2

latent_dir = '/mnt/sda1/saksham/TI2AV/others/ltx_lora_training_i2v_t2v/cache_121x768x512'
vid_dir = '/mnt/sda1/saksham/TI2AV/others/AVSync15/videos_together'
save_frame_dir = '/mnt/sda1/saksham/TI2AV/others/ltx_lora_training_i2v_t2v/first_frame'

files = os.listdir(latent_dir)

def save_first_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_image_path, frame)
        print(f"First frame saved at: {output_image_path}")
    else:
        print("Failed to read the video file.")
    cap.release()

def save_caption(caption, save_path):
    with open(save_path, 'w') as f:
        json.dump(caption, f)

for file in tqdm(files):
    vid_path = os.path.join(vid_dir, file.replace('.pt', '.mp4'))
    save_path = os.path.join(save_frame_dir, file.replace('.pt', '.jpg'))
    save_first_frame(vid_path, save_path)

    latent_path = os.path.join(latent_dir, file)
    latent = torch.load(latent_path)
    caption = latent['captions'][0]
    
    save_caption_path = os.path.join(save_frame_dir, file.replace('.pt', '.txt'))
    save_caption(caption, save_caption_path)