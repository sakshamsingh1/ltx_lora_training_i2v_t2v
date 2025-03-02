import librosa
import pandas as pd
from tqdm import tqdm
import json
import os
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)
model.to("cuda")

def get_captions(audio_path):
    prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate a very detailed caption in English:"
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(text=prompt, audios=audio, return_tensors="pt",sampling_rate=sr).to("cuda")
    max_len = max(256, inputs.feature_attention_mask.size(1))
    generated_ids = model.generate(**inputs, max_length=max_len)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

meta_path = '/mnt/ssd0/saksham/i2av/AVSync15/metadata.csv'
aud_dir = '/mnt/ssd0/saksham/i2av/AVSync15/audios_together'
save_json = '/mnt/ssd0/saksham/i2av/AVSync15/aud_caption.json'
DRY_RUN = False

df = pd.read_csv(meta_path)
aud_caption = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    aud_path = os.path.join(aud_dir, row['vid'] + '.wav')
    caption = get_captions(aud_path)
    aud_caption[row['vid']] = caption
    if DRY_RUN:
        print(row['vid'], caption)
        if i == 5:
            break

if not DRY_RUN:
    with open(save_json, 'w') as f:
        json.dump(aud_caption, f)