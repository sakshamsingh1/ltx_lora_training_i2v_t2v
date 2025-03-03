# create videos together
# import shutil
# import os
# import pandas as pd
# from tqdm import tqdm

# meta_path = '/mnt/ssd0/saksham/i2av/AVSync15/metadata.csv'
# base_read_dir = '/home/sxk230060/TI2AV/data/AVSync15/videos'
# save_dir = '/mnt/ssd0/saksham/i2av/AVSync15/videos_together'

# df = pd.read_csv(meta_path)

# for i, row in tqdm(df.iterrows()):
#     video_path = os.path.join(base_read_dir, row['label'], row['vid'] + '.mp4')
#     save_path = os.path.join(save_dir, row['vid'] + '.mp4')
#     shutil.copy(video_path, save_path)

############################################################################
# extract audio
# import os
# import pandas as pd
# from tqdm import tqdm
# import librosa
# from moviepy import VideoFileClip
# import soundfile as sf

# meta_path = '/mnt/ssd0/saksham/i2av/AVSync15/metadata.csv'
# base_vid_dir = '/mnt/ssd0/saksham/i2av/AVSync15/videos_together'
# save_aud_dir = '/mnt/ssd0/saksham/i2av/AVSync15/audios_together'
# target_sr = 16000

# def extract_audio(video_path, output_audio_path):
#     video = VideoFileClip(video_path)
#     audio = video.audio
#     audio.write_audiofile(output_audio_path)

#     # Resample to target_sr and mono
#     audio, sr = librosa.load(output_audio_path, sr=target_sr, mono=True)  # Convert to mono & resample
#     sf.write(output_audio_path, audio, target_sr)

# df = pd.read_csv(meta_path)
# for i, row in tqdm(df.iterrows(), total=len(df)):
#     video_path = os.path.join(base_vid_dir, row['vid'] + '.mp4')
#     save_path = os.path.join(save_aud_dir, row['vid'] + '.wav')
#     extract_audio(video_path, save_path)    

############################################################################
# extract first frame    
import os
from tqdm import tqdm
import cv2
import pandas as pd

meta_path='/home/sxk230060/TI2AV/data/AVSync15/metadata.csv'
df = pd.read_csv(meta_path)

vid_dir = '/mnt/ssd0/saksham/i2av/AVSync15/videos_together'
save_frame_dir = '/mnt/ssd0/saksham/i2av/AVSync15/first_frame'

def save_first_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_image_path, frame)
    else:
        print("Failed to read the video file.")
    cap.release()

for i, row in tqdm(df.iterrows(), total=len(df)):
    vid_path = os.path.join(vid_dir, row['vid']+ '.mp4')
    save_path = os.path.join(save_frame_dir, row['vid'] + '.jpg')
    save_first_frame(vid_path, save_path)