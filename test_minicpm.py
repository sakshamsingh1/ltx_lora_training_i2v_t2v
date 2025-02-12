

import ollama
import base64
import decord
import numpy as np
import io
from torchvision import transforms

def pil_to_base64(image):
  byte_stream = io.BytesIO()
  image.save(byte_stream, format='JPEG')
  byte_stream.seek(0)
  return base64.b64encode(byte_stream.read()).decode('utf-8')

video_file = './data/test.mp4'
video_reader = decord.VideoReader(video_file)
decord.bridge.set_bridge("torch")
video = video_reader.get_batch(
    np.linspace(0,  len(video_reader) - 1, 8).astype(np.int_)
).byte()

print("video", video.shape)

frames = [transforms.ToPILImage()(image.permute(2, 0, 1)).convert("RGB").resize((640, 360)) for image in video]
# frames[0].save("data/test_frame.jpg")

images = [ pil_to_base64(image) for image in frames]
print("images", len(images))

# with open("data/base64_test.png", "wb") as fh:
#   fh.write(base64.decodebytes(bytes(images[0], "utf-8")))
# with open("data/puppy.jpg", "rb") as f:
#   b64_image = base64.b64encode(f.read()).decode("utf-8")

# prompt = """
# Please describe the video in short, focusing on:
# - The viewing angle of the character (front view, side view, back view, or other angles)
# - The character's position and orientation in the scene
# - The character's movements and actions
# - Any changes in the character's viewing angle during the video
# - Other important visual details
# Please be specific about the viewing perspective when describing the character.
# """

prompt = """describe this video in this order: camera angle, main subject, make the description short
"""

client = ollama.Client()
response = client.chat(
    model="minicpm-v:8b-2.6-q5_0",
    messages=[{
      "role":"user",
      "content": prompt, # "describe this video in short",
      "images": images }
    ]
)
print(response["message"]["content"])