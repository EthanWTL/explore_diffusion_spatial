import torch
from diffusers import CogVideoXPipeline

pipe = CogVideoXPipeline.from_pretrained(
    "/project/sds-rise/ethan/explore_diffusion_spatial/huggingface/models/CogVideoX-2b"
)

pipe.to(dtype = torch.float16)

new_layer = torch.nn.Conv2d(32, 1920, kernel_size=(2, 2), stride=(2, 2)).to(dtype=torch.float16)

duplicated_weights = pipe.transformer.patch_embed.proj.weight.repeat(1, 2, 1, 1)
duplicated_bias = pipe.transformer.patch_embed.proj.bias

with torch.no_grad():
    new_layer.weight.copy_(duplicated_weights)
    new_layer.bias.copy_(duplicated_bias)

pipe.transformer.patch_embed.proj = new_layer
pipe.transformer.register_to_config(in_channels=32)

pipe.to(dtype = torch.float16)

pipe.save_pretrained("/project/sds-rise/ethan/explore_diffusion_spatial/huggingface/models/CogVideoX-2b-32in-16out")