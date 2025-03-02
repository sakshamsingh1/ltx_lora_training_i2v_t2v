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

# extract audio
import os
import pandas as pd
from tqdm import tqdm
import librosa
from moviepy import VideoFileClip
import soundfile as sf

meta_path = '/mnt/ssd0/saksham/i2av/AVSync15/metadata.csv'
base_vid_dir = '/mnt/ssd0/saksham/i2av/AVSync15/videos_together'
save_aud_dir = '/mnt/ssd0/saksham/i2av/AVSync15/audios_together'
target_sr = 16000

def extract_audio(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)

    # Resample to target_sr and mono
    audio, sr = librosa.load(output_audio_path, sr=target_sr, mono=True)  # Convert to mono & resample
    sf.write(output_audio_path, audio, target_sr)

df = pd.read_csv(meta_path)
for i, row in tqdm(df.iterrows(), total=len(df)):
    video_path = os.path.join(base_vid_dir, row['vid'] + '.mp4')
    save_path = os.path.join(save_aud_dir, row['vid'] + '.wav')
    extract_audio(video_path, save_path)    