import os, random, math, time
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import torch
from torchvision import transforms
from PIL import Image
import gc

os.environ['all_proxy']=''
os.environ['all_proxy']=''

# import ollama
import base64
import io
from icecream import ic
from constants import DEFAULT_HEIGHT_BUCKETS, DEFAULT_WIDTH_BUCKETS, DEFAULT_FRAME_BUCKETS
# if there is a memery leak in the code, we'll shut it down manually
import psutil

ic.disable()

memusage = psutil.virtual_memory()[2]
assert memusage < 85, "Impending memory leak, memory needs to be cleared before running."

def get_frames(inp: str, w: int, h: int, start_sec: float = 0, duration: float = None, f: int = None, fps = None) -> np.ndarray:
    args = []
    if duration is not None:
        args += ["-t", f"{duration:.2f}"]
    elif f is not None:
        args += ["-frames:v", str(f)]
    if fps is not None:
        args += ["-r", str(fps)]
    
    args = ["ffmpeg", "-nostdin", "-ss", f"{start_sec:.2f}", "-i", inp, *args, 
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "pipe:"]
    
    process = subprocess.Popen(args, stderr=-1, stdout=-1)
    out, err = process.communicate()
    retcode = process.poll()
    if retcode:
        raise Exception(f"{inp}: ffmpeg error: {err.decode('utf-8')}")

    process.terminate()
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3) # b, h, w, c

class Captioner():
    def __init__(self, model="minicpm-v:8b-2.6-q5_0", prompt=None):
        self.model = model
        default_prompt = "describe this video in short"
        self.prompt = prompt or default_prompt
        
        start = ["The", "This"]
        kind = ["video", "image", "scene", "animated sequence"]
        act = ["displays", "shows", "features", "is", "depicts", "presents", "showcases", "captures" ]
        
        bad_phrese = []
        for ss in start:
            for kk in kind:
                for aa in act:
                    bad_phrese.append(f"{ss} {kk} {aa}")
                    
        self.should_remove_phrese=[
            "In the video",
        ] + bad_phrese
        
    @staticmethod
    def pil_to_base64(image):
      byte_stream = io.BytesIO()
      image.save(byte_stream, format='JPEG')
      byte_stream.seek(0)
      return base64.b64encode(byte_stream.read()).decode('utf-8')
    
    def remove_phrese(self, cap):
        # only keep the primary part of the caption
        cap = cap.replace("\n\n", "\n")
        
        for ii in self.should_remove_phrese:
            cap = cap.replace(ii, "")
            
        return cap
        
    def get_caption(self, frames, size=(640, 320), frame_skip=2):
        self.client = ollama.Client()

        # 24fps to 8fps
        frames = frames[::frame_skip]
        if isinstance(frames, np.ndarray):
            frames = [Image.fromarray(image).convert("RGB").resize(size) for image in frames]
        else:
            frames = [transforms.ToPILImage()(image.permute(2, 0, 1)).convert("RGB").resize(size) for image in frames]
        images = [ self.pil_to_base64(image) for image in frames]
        
        response = self.client.chat(
            model=self.model,
            keep_alive="6s",
            options=dict(num_predict=128),
            messages=[{
              "role":"user",
              "content": self.prompt, # "describe this video in short",
              "images": images }
            ]
        )
        cap = response["message"]["content"]
        return self.remove_phrese(cap)
        
class VideoFramesDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        video_dir, # list or string path
        video_list_file: str,
        cache_dir: str,
        width: int = 1024,
        height: int = 576,
        num_frames: int = 49, 
        fps: int = 24,
        # to filter out short clips
        get_frames_max: int = 50 * 24, # prevent super long videos
        # cach_frames_min: int = 9,
        prompt_prefix = "freeze time, camera orbit left,",
    ):
        super().__init__()
        assert width in DEFAULT_WIDTH_BUCKETS, f"width only supported in: {DEFAULT_WIDTH_BUCKETS}"
        assert height in DEFAULT_HEIGHT_BUCKETS, f"height only supported in: {DEFAULT_HEIGHT_BUCKETS}"
        assert num_frames in DEFAULT_FRAME_BUCKETS, f"frames should in: {DEFAULT_FRAME_BUCKETS}"
        
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.video_dir = video_dir
        
        self.cache_dir = Path(f"{cache_dir}_{num_frames}x{width}x{height}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.get_frames_max = get_frames_max
        self.prompt_prefix = prompt_prefix
        
        self.videos = []
        self.data = []

        video_list_file = os.path.join(video_dir, video_list_file)
        with open(video_list_file, "r") as f:
            self.videos = [os.path.join(video_dir, line.strip()) for line in f.readlines()]

        # if isinstance(video_dir, str):
        #     self.videos = self.load_videos(video_dir)
        # else:
        #     for dd in video_dir:
        #         self.videos += self.load_videos(dd)
        
        print(f"{type(self).__name__} found {len(self.videos)} videos ")

    # def load_videos(self, dirname):
    #     videos = []
    #     for root, dirs, files in os.walk(dirname):
    #         for file in files:
    #             if (file.endswith('.mp4') or file.endswith('.mov')) and file[0] != ".":
    #                 videos.append(os.path.join(root, file))
    #     assert len(videos) > 0, "目标文件夹内没有视频文件"
    #     return videos
    
    def to_tensor(self, data, device="cuda", dtype=torch.bfloat16):
        input = (data / 255) * 2.0 - 1.0
        # from (t, h, w, c) to b (t  c, h, w)
        return torch.from_numpy(input).permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype=dtype)

    def cache_frames(self, to_latent, to_caption, to_embedding, device="cuda"):
        print(f"building caches, video count: {len(self.videos)}")

        for ii, vid in enumerate(tqdm(self.videos)):
            dest = os.path.join(self.cache_dir, os.path.basename(vid).rsplit(".", 1)[0] + ".pt")
            ic(dest)
            if os.path.exists(dest):
                # print("skip:", dest)
                continue
            try:
                video_frames = get_frames(vid, self.width, self.height, 0,  f=self.get_frames_max, fps=self.fps)
            except Exception as e:
                print("error file:", vid, e)
                continue
            # remove first frame for frame blending motion blured video
            video_frames = video_frames[1:]

            if len(video_frames) < self.num_frames:
                continue
            # divid into parts
            iters = len(video_frames) // self.num_frames
            latents = []
            embedds = []
            masks = []
            captions = []
            infos = []
            first_frames = []

            for idx in range(iters):
                frames = video_frames[ idx*self.num_frames : (idx + 1) * self.num_frames ]
                ic(frames.shape) 
                # caption = self.prompt_prefix + " " + to_caption(frames)
                caption = self.prompt_prefix + " " + to_caption(vid)
                ic(caption)
                embedding, mask = to_embedding(caption.replace("  ", " "))
                ic(embedding.shape, mask.shape)

                frames_t = self.to_tensor(frames, device=device)
                ic(frames_t)
                latent, num_frames, height, width = to_latent(frames_t)
                assert latent.ndim == 3, "patched latent should have 3 dims"
                ic(latent.shape, latent)
                ff, _, _, _ = to_latent(frames_t[:, 0:1, :, :, :])
                ic(ff)
                # make sure is not nan
                assert not torch.isnan(ff.flatten()[0]), "nan encountered! abort!"
                
                captions.append(caption)
                embedds.append(embedding)
                masks.append(mask)
                latents.append(latent)
                first_frames.append(ff)
                infos.append(dict(num_frames=num_frames, height=height, width=width))

            latents = torch.cat(latents, dim=0)
            embedds = torch.cat(embedds, dim=0)
            masks = torch.cat(masks, dim=0)
            
            # print(latent.shape, latent_lr.shape)
            # np.savez(dest, hr=latent, lr=latent_lr)
            torch.save(dict(latents=latents, 
              embedds=embedds, 
              masks=masks, 
              captions=captions, 
              first_frames=first_frames,
              meta_info=infos), dest)
            self.data.append(dest)

            # Force garbage collection
            # gc.collect()
            # if ii % 20 == 0:
            #     time.sleep(7) # wait for ollama to clean up

            memusage = psutil.virtual_memory()[2]
            assert memusage < 80, "即将内存泄漏，强行关闭，请重新启动进程"
            
        print(f">> cached {len(self.data)} videos")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.videos[idx]

class Cog_Captions():
    def __init__(self):
        self.cog_caption_path = '/mnt/sda1/saksham/TI2AV/others/AVSync15/cog_train_captions.json'
        self.caption_map = json.load(open(self.cog_caption_path, "r"))
        
    def get_caption(self, vid):
        label, vid = vid.split("/")[-2:]
        label = label.replace("__", " ").replace("_", " ")
        vid = vid.rsplit(".", 1)[0]
        text_caption = label + ", " + self.caption_map[vid]
        return text_caption

if __name__ == "__main__":
    import argparse
    from yaml import load, Loader
    from tqdm import tqdm

    from ltx_video_lora import *
    
    video_base_dir = "/mnt/sda1/saksham/TI2AV/others/AVSync15/videos"
    video_list_file = "train_filter.txt"
    cache_dir = "/mnt/sda1/saksham/TI2AV/others/ltx_lora_training_i2v_t2v/cache"
    
    config_file = "./configs/ltx.yaml"
    device = "cuda"
    dtype = torch.bfloat16
    prompt_prefix = "sounding object, "
    # ------------------- 

    config_dict = load(open(config_file, "r"), Loader=Loader)
    args = argparse.Namespace(**config_dict)

    # ----------- prepare models -------------
    dataset = VideoFramesDataset(
        video_dir=video_base_dir,
        video_list_file=video_list_file,
        cache_dir=cache_dir,
        width=768,
        height=512,
        num_frames=121,
        prompt_prefix=prompt_prefix,
    )

    captioner = Cog_Captions()
    cond_models = load_condition_models()
    tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
    text_encoder = text_encoder.to(device, dtype=dtype)
    vae = load_latent_models()["vae"].to(device, dtype=dtype)

    def to_latent(frames_tensor):
        # vaedtype = next(vae.parameters()).dtype
        ic(frames_tensor.shape)
        assert frames_tensor.size(2) == 3, f"frames should be in shape: (b, f, c, h, w) provided: {frames_tensor.shape}"
        with torch.no_grad():
            data = prepare_latents(
                    vae=vae,
                    image_or_video=frames_tensor,
                    device=device,
                    dtype=dtype,
                )
        return data["latents"].cpu().to(dtype), data["num_frames"], data["height"], data["width"]
            
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
    
    dataset.cache_frames(to_latent, captioner.get_caption, to_embedding)
    print("done!")


