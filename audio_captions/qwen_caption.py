import os
import pandas as pd
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import json

meta_path = '/mnt/sda1/saksham/TI2AV/others/AVSync15/metadata.csv'
aud_base_dir = '/mnt/sda1/saksham/TI2AV/others/AVSync15/audios'
save_path = '/mnt/sda1/saksham/TI2AV/others/AVSync15/audio_caption.json'
SAMPLE_RATE = 16000
DRY_RUN=False

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
df = pd.read_csv(meta_path)

def get_aud_caption(aud_path):
    conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": aud_path},
        {"type": "text", "text": "What can you hear? Give me a detailed description of the sounds you hear."},
    ]},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append( librosa.load(ele['audio_url'], sr=SAMPLE_RATE)[0] )

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=SAMPLE_RATE)
    inputs.input_ids = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_length=352)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

vid_caption = {}
for idx, row in df.iterrows():
    aud_path = os.path.join(aud_base_dir, row['vid'] + '.wav')
    curr_caption = get_aud_caption(aud_path)
    vid_caption[row['vid']] = curr_caption

    if DRY_RUN:
        print(f"{row['vid']}: {curr_caption}")
        if idx > 5:
            break

json.dump(vid_caption, open(save_path, 'w'))