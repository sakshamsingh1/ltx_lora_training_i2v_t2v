import os
import pandas as pd
import librosa
import soundfile as sf
from moviepy import VideoFileClip

meta_path = '/mnt/sda1/saksham/TI2AV/others/AVSync15/metadata.csv'
df = pd.read_csv(meta_path)

def extract_audio_moviepy(video_path, output_audio_path, target_sr=16000):
    video = VideoFileClip(video_path)
    audio_path_temp = "temp_audio.wav"  # Temporary file
    video.audio.write_audiofile(audio_path_temp, codec="pcm_s16le")
    audio, sr = librosa.load(audio_path_temp, sr=target_sr, mono=True)
    sf.write(output_audio_path, audio, target_sr)

vid_dir = '/mnt/sda1/saksham/TI2AV/others/AVSync15/videos_together'
save_dir = '/mnt/sda1/saksham/TI2AV/others/AVSync15/audios'

for idx, row in df.iterrows():
    vid_path = os.path.join(vid_dir, row['vid'] + '.mp4')
    aud_path = os.path.join(save_dir, row['vid'] + '.wav')
    extract_audio_moviepy(vid_path, aud_path)