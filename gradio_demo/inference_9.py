import time
import sys
sys.path.append('./')
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL

import torch
import os
import numpy as np
from torchvision import transforms
import argparse
import pynvml

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def print_memory_usage(step_name=""):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 取得 GPU 0
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Windows 工作管理員顯示 VRAM: {info.used / 1024 ** 3:.1f} GB")


def pil_to_binary_mask(pil_image, threshold=0):
    grayscale_image = np.array(pil_image.convert("L"))
    mask = (grayscale_image > threshold).astype(np.uint8) * 255
    output_mask = Image.fromarray(mask)
    return output_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a Inference script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default= "9channel retrain/checkpoint-61360",required=False,)
    parser.add_argument("--width",type=int,default=768,)
    parser.add_argument("--height",type=int,default=1024,)
    parser.add_argument("--num_inference_steps",type=int,default=10,)
    parser.add_argument("--data_path",type=str,default= "../")
    parser.add_argument("--input_name", type=str, default="1", )
    parser.add_argument("--save_name",type=str,default="inference.png",)
    parser.add_argument("--seed", type=int, default=42,)
    arg = parser.parse_args()
    return arg


print_memory_usage()
args = parse_args()
base_path = 'C://Users/cgal/Desktop/' + args.pretrained_model_name_or_path
resolution = [args.width, args.height]

example_path = os.path.join(os.path.dirname(__file__), 'example')

def init():
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
        local_files_only=True
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_path,
        subfolder="scheduler",
        local_files_only=True
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
        local_files_only=True
        )
    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=torch.float16,
        local_files_only=True
    )

    # "stabilityai/stable-diffusion-xl-base-1.0",
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
        local_files_only=True
    )

    UNet_Encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    tensor_transfrom = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
        )

    pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder
    return pipe, tensor_transfrom


pipe, tensor_transform = init()
pipe.to(device)


def inference():
    seed_ = args.seed
    start_time = time.time()

    a = Image.open(f"{args.data_path}data/cloth/{args.input_name}.jpg")
    b = Image.open(f"{args.data_path}data/image/{args.input_name}.jpg")
    c = Image.open(f"{args.data_path}data/mask/{args.input_name}.png")

    garm_img = a.convert("RGB").resize((resolution[0], resolution[1]))
    h_orig = b.convert("RGB")
    h_img = h_orig.resize((resolution[0], resolution[1]))
    mask_img = pil_to_binary_mask(c.convert("RGB").resize((resolution[0], resolution[1])))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                with torch.inference_mode():
                    prompt_embeds = torch.zeros(1, 77, 2048)
                    negative_prompt_embeds = torch.zeros(1, 77, 2048)
                    pooled_prompt_embeds = torch.zeros(1, 1280)
                    negative_pooled_prompt_embeds = torch.zeros(1, 1280)
                    prompt_embeds_c = torch.zeros(1, 77, 2048)

                    garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
                    generator = torch.Generator(device).manual_seed(seed_) if seed_ is not None else None
                    x_star = pipe(
                        prompt_embeds=prompt_embeds.to(device, torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        strength=1.0,
                        text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                        cloth=garm_tensor,
                        mask_image=mask_img,
                        image=h_img,
                        height=resolution[1],
                        width=resolution[0],
                        ip_adapter_image=garm_img.resize((resolution[0], resolution[1])),
                        guidance_scale=2.0,
                    )[0]
                    os.makedirs(f'{args.data_path}result', exist_ok=True)
                    x_star[0].save(f'{args.data_path}result/{args.save_name}')

    print(f"Inference time:{time.time() - start_time}")
    # torch.cuda.empty_cache()


inference()
inference()
