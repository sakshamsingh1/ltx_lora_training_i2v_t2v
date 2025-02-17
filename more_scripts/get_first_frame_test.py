import os
import pandas as pd
import json
from tqdm import tqdm
import cv2

meta_path = '/mnt/sda1/saksham/TI2AV/others/AVSync15/metadata.csv'
vid_dir = '/mnt/sda1/saksham/TI2AV/others/AVSync15/videos_together'
save_frame_dir = '/mnt/sda1/saksham/TI2AV/others/ltx_lora_training_i2v_t2v/first_frame_test'
caption_path = '/mnt/sda1/saksham/TI2AV/AVSync15/cog_test_caption.json'

os.makedirs(save_frame_dir, exist_ok=True)

df = pd.read_csv(meta_path)
df = df[df['split'] == 'test'].reset_index(drop=True)
PREFIX = "sounding object, "

caption_map = json.load(open(caption_path)) 

def save_first_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_image_path, frame)
        # print(f"First frame saved at: {output_image_path}")
    else:
        print("Failed to read the video file.")
    cap.release()

def save_caption(caption, save_path):
    with open(save_path, 'w') as f:
        json.dump(caption, f)

for i, row in tqdm(df.iterrows(), total=len(df)):
    vid = row['vid']
    vid_path = os.path.join(vid_dir, vid + '.mp4')
    save_path = os.path.join(save_frame_dir, vid + '.jpg')
    save_first_frame(vid_path, save_path)

    label = row['label']
    label = label.replace("__", " ").replace("_", " ")
    text_caption = PREFIX + label + ", " + caption_map[vid]

    save_caption_path = os.path.join(save_frame_dir, vid + '.txt')
    save_caption(text_caption, save_caption_path)