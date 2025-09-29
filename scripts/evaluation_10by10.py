from diffusers import CogVideoXImageToVideoPipeline
import torch
from diffusers.utils import export_to_video
from torch.utils.data import DataLoader

from CogVideo.finetune.dataloader.maze_dataset import build_all_loaders


TRAIN_JSONL = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/10by10/train/info_labels.jsonl"
TRAIN_IMG   = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/10by10/train/images"
TRAIN_PRM   = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/10by10/train/prompts.txt"

VAL_JSONL   = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/10by10/val/info_labels.jsonl"
VAL_IMG     = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/10by10/val/images"
VAL_PRM     = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/10by10/val/prompts.txt"

TEST_JSONL  = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/10by10/test/info_labels.jsonl"
TEST_IMG    = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/10by10/test/images"
TEST_PRM    = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/10by10/test/prompts.txt"

(train_ds, train_dl), (val_ds, val_dl), (test_ds, test_dl) = build_all_loaders(
    train_jsonl=TRAIN_JSONL, train_images=TRAIN_IMG, train_prompts=TRAIN_PRM,
    val_jsonl=VAL_JSONL,     val_images=VAL_IMG,     val_prompts=VAL_PRM,
    test_jsonl=TEST_JSONL,   test_images=TEST_IMG,   test_prompts=TEST_PRM,
    batch_size=1, num_workers=0,
    num_frames=49, assert_geometry_match=True, shuffle_train=True
)

test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=False)

pipe = CogVideoXImageToVideoPipeline.from_pretrained("/project/sds-rise/ethan/explore_diffusion_spatial/huggingface/models/CogVideoX-2b-32in-16out-10by10")

pipe.to(device="cuda:0", dtype=torch.float16)

height = 480
width=720
num_videos_per_prompt = 1
num_inference_steps = 50
num_frames = 49
guidance_scale = 6.0
seed=42
fps=16

for batch in test_dl:
    index = batch['index']
    prompt = batch["prompt"]
    image = batch['image']

    output_path = f"/project/sds-rise/ethan/explore_diffusion_spatial/evaluation/artifacts/10by10/basic/maze_{index}.mp4"

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
