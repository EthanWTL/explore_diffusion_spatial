from diffusers import CogVideoXImageToVideoPipeline
import torch
from diffusers.utils import export_to_video, load_image
from torch.utils.data import DataLoader

from CogVideo.finetune.dataloader.maze_dataset import build_all_loaders

prompts = []
with open("/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/2path/prompts.txt", 'r') as file:
    for line in file:
        prompts.append(line.strip()) 


pipe = CogVideoXImageToVideoPipeline.from_pretrained("/project/sds-rise/ethan/explore_diffusion_spatial/huggingface/models/CogVideoX-2b-32in-16out-5by5")

pipe.to(device="cuda:0", dtype=torch.float16)

num_plan = len(prompts)
height = 480
width=720
num_videos_per_prompt = 1
num_inference_steps = 50
num_frames = 49
guidance_scale = 6.0
seed=42
fps=16

index = 7

prompt = prompts[index]
image = load_image(image=f"/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/2path/images/maze_{index:05d}.png")
output_path = f"/project/sds-rise/ethan/explore_diffusion_spatial/evaluation/artifacts/5by5/2path/maze_{index:05d}.mp4"

video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]

export_to_video(video_generate, output_path, fps=fps)

    
