import time
import sys
sys.path.append('./')
from PIL import Image
import gradio as gr
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
import gc
import numpy as np
from torchvision import transforms
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
import pynvml
from src.transformerhacked_tryon import CustomIdentity
from src.attentionhacked_tryon import BasicTransformerBlock
from safetensors.torch import load_file


def print_memory_usage(step_name=""):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 取得 GPU 0
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Windows 工作管理員顯示 VRAM: {info.used / 1024 ** 3:.1f} GB")


print_memory_usage("-")
ch = time.time()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
resolution = [720, 1280]
check_use_opt = False


def switch_use_opt(isTurnOn):
    global check_use_opt
    check_use_opt = isTurnOn


def change_resolution(option):
    global resolution
    if option == "720x1280":
        resolution = [720, 1280]
    else:
        resolution = [360, 640]


def pil_to_binary_mask(pil_image, threshold=0):
    grayscale_image = np.array(pil_image.convert("L"))
    mask = (grayscale_image > threshold).astype(np.uint8) * 255
    output_mask = Image.fromarray(mask)
    return output_mask


# base_path = 'yisol/IDM-VTON'
# base_path = 'C://Users/cgal/Desktop/distillation2/checkpoint-6280'
# remove_attention_index = [6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 39, 40, 41, 46, 47, 48, 49, 50, 51, 52, 53, 54, 58, 60]
remove_attention_index = []
base_path = 'C://Users/cgal/Desktop/9channel retrain/checkpoint-61360'
example_path = os.path.join(os.path.dirname(__file__), 'example')
k = 0


def is_prunable(module, name):
    return isinstance(module, BasicTransformerBlock)


def replace_prunable_layers(module):
    # module 是 try on Net
    for name, child in module.named_children():
        # 如果子模塊本身是可剪枝的，就替換
        if is_prunable(child, name):
            print(f"Replacing {name} of type {type(child)} with Identity")
            remove_attn_layer(module, name)
        else:
            # 否則遞歸處理子模塊
            replace_prunable_layers(child)


def remove_layer(model, layer_name):
    setattr(model, layer_name, CustomIdentity())


def remove_attn_layer(model, layer_name):
    global k
    k += 1
    # 生成剪枝模型，將指定層替換為 Identity（跳過該層）
    if k in remove_attention_index:
        print("remove")
        remove_layer(model, layer_name)
    return


def init():
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    if len(remove_attention_index) != 0:
        replace_prunable_layers(unet)
        unet.load_state_dict(load_file(f"{base_path}/unet/diffusion_pytorch_model.safetensors"))
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
# print(pipe.unet)
# print()
# print(pipe.unet_encoder)
# for name, module in pipe.unet_encoder.named_children():
#     if name == "up_blocks" or name == "down_blocks":
#         for na, sm in module.named_children():
#             for nam, sm1 in sm.named_children():
#                 for final, sss in sm1.named_children():
#                     total = 0
#                     for names, param in sss.named_parameters():
#                         num_params = param.numel()  # 參數數量
#                         total += num_params
#                     print(f"{nam}: {total}")
# to = 0
# for name, param in pipe.unet_encoder.named_parameters():
#     num_params = param.numel()  # 參數數量
#     to += num_params
# print(f"garment unet: {to}")
# print("")
# for name, module in pipe.unet.named_children():
#     if name == "up_blocks" or name == "down_blocks":
#         for na, sm in module.named_children():
#             for nam, sm1 in sm.named_children():
#                 for final, sss in sm1.named_children():
#                     total = 0
#                     for names, param in sss.named_parameters():
#                         num_params = param.numel()  # 參數數量
#                         total += num_params
#                     print(f"{nam}: {total}")
#     else:
#         k = 0
#         for names, param in module.named_parameters():
#             num_params = param.numel()
#             k += num_params
#         print(f"{name}: {k}")

# to = 0
# for name, param in pipe.unet.named_parameters():
#     num_params = param.numel()  # 參數數量
#     to += num_params
# print(f"tryon unet: {to}")
pipe.to(device)
print_memory_usage("load")

def clean():
    torch.cuda.empty_cache()


def show_device():
    if pipe.unet is not None:
        d_unet = next(pipe.unet.parameters()).device
        print(f'UNet is on: {d_unet}')

    if pipe.text_encoder is not None:
        d_text_encoder = next(pipe.text_encoder.parameters()).device
        print(f'Text Encoder 1 is on: {d_text_encoder}')

    if pipe.text_encoder_2 is not None:
        d_text_encoder_2 = next(pipe.text_encoder_2.parameters()).device
        print(f'Text Encoder 2 is on: {d_text_encoder_2}')

    if pipe.image_encoder is not None:
        d_image_encoder = next(pipe.image_encoder.parameters()).device
        print(f'Image Encoder is on: {d_image_encoder}')

    if pipe.vae is not None:
        d_vae = next(pipe.vae.parameters()).device
        print(f'VAE is on: {d_vae}')


def move_model(is_move2cpu, model):
    if model == 'unet':
        if pipe.unet is not None:
            pipe.unet.to('cpu' if is_move2cpu else 'cuda:0')

    elif model == 'text_encoder':
        if pipe.text_encoder is not None:
            pipe.text_encoder.to('cpu' if is_move2cpu else 'cuda:0')
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to('cpu' if is_move2cpu else 'cuda:0')

    elif model == 'image_encoder':
        if pipe.image_encoder is not None:
            pipe.image_encoder.to('cpu' if is_move2cpu else 'cuda:0')

    elif model == 'vae':
        if pipe.vae is not None:
            pipe.vae.to('cpu' if is_move2cpu else 'cuda:0')

    elif model == 'unet_encoder':
        if pipe.unet_encoder is not None:
            pipe.unet_encoder.to('cpu' if is_move2cpu else 'cuda:0')
    torch.cuda.empty_cache()


def gamma_curve_optimization(n, gamma):
    t = [i / (n - 1) for i in range(n)]
    t = t[::-1]
    tl = 1
    tu = 1 + int(1000 / n) * (n - 1)
    t = [int((tu-tl) * (t_val ** gamma) + 1) for t_val in t]
    return t


def start_tryon(dict_, garm_img_, denoise_steps_, seed_, gamma_):
    print_memory_usage("初始化")
    gs = gamma_curve_optimization(denoise_steps_, gamma_ ** -1)
    start_time = time.time()

    garm_img_ = garm_img_.convert("RGB").resize((resolution[0], resolution[1]))

    human_img_orig = dict_["background"].convert("RGB")
    human_img = human_img_orig.resize((resolution[0], resolution[1]))

    mask = pil_to_binary_mask(dict_['layers'][0].convert("RGB").resize((resolution[0], resolution[1])))
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)
    mask_time = time.time()
    print(f"preprocess-mask: {mask_time - start_time:.2f} s")
    #
    # pose_img = Image.new('RGB', (resolution[0], resolution[1]), (0, 0, 0))
    #
    pose_time = time.time()
    # print(f"preprocess-pose: {pose_time - mask_time:.2f} s")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                with torch.inference_mode():
                    new_time = time.time()
                    print(f"check time:{new_time - pose_time:.2f} s")
                    prompt_embeds = torch.zeros(1, 77, 2048, device=device)
                    negative_prompt_embeds = torch.zeros(1, 77, 2048, device=device)
                    pooled_prompt_embeds = torch.zeros(1, 1280, device=device)
                    negative_pooled_prompt_embeds = torch.zeros(1, 1280, device=device)
                    prompt_embeds_c = torch.zeros(1, 77, 2048, device=device)
                    encode_prompt_time = time.time()
                    print(f"preprocess-encode_prompt: {encode_prompt_time - new_time:.2f} s")

                    # pose_img = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
                    garm_tensor = tensor_transform(garm_img_).unsqueeze(0).to(device, torch.float16)
                    generator = torch.Generator(device).manual_seed(seed_) if seed_ is not None else None
                    move_time = time.time()
                    print(f"move to gpu: {move_time - encode_prompt_time:.2f} s")
                    print_memory_usage("move to gpu")
                    if check_use_opt:
                        images = pipe(
                            timesteps=gs,
                            prompt_embeds=prompt_embeds.to(torch.float16),
                            negative_prompt_embeds=negative_prompt_embeds.to(torch.float16),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(torch.float16),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(torch.float16),
                            generator=generator,
                            strength=1.0,
                            text_embeds_cloth=prompt_embeds_c.to(torch.float16),
                            cloth=garm_tensor,
                            mask_image=mask,
                            image=human_img,
                            height=resolution[1],
                            width=resolution[0],
                            ip_adapter_image=garm_img_.resize((resolution[0], resolution[1])),
                            guidance_scale=2.0,
                        )[0]
                    else:
                        images = pipe(
                            prompt_embeds=prompt_embeds.to(torch.float16),
                            negative_prompt_embeds=negative_prompt_embeds.to(torch.float16),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(torch.float16),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(torch.float16),
                            num_inference_steps=denoise_steps_,
                            generator=generator,
                            strength=1.0,
                            text_embeds_cloth=prompt_embeds_c.to(torch.float16),
                            cloth=garm_tensor,
                            mask_image=mask,
                            image=human_img,
                            height=resolution[1],
                            width=resolution[0],
                            ip_adapter_image=garm_img_.resize((resolution[0], resolution[1])),
                            guidance_scale=2.0,
                        )[0]

    print(f"total: {time.time() - start_time:.2f} s")
    clean()
    print_memory_usage("final")
    return images[0], mask_gray


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen', interactive=True)

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            # with gr.Row(elem_id="prompt-container"):
            #     with gr.Row():
            #         prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img",show_share_button=False)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            image_out = gr.Image(label="Output", elem_id="output-img",show_share_button=False)




    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=5, maximum=50, value=10, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)
                gamma = gr.Number(label="gamma", minimum=1, maximum=10, step=0.1, value=2)
    radio = gr.Radio(choices=["720x1280", "360x640"], label="Resolution", value="720x1280")
    with gr.Row():
        show_button = gr.Button(value="Show Device")
        clean_button = gr.Button(value="Clean GPU")
        use_opt = gr.Checkbox(label="use opt(false = init)", value=False)
    with gr.Row():
        switch_u = gr.Checkbox(label="unet on cpu", value=False)
        switch_te = gr.Checkbox(label="text_encoder on cpu", value=False)
        switch_ie = gr.Checkbox(label="image_encoder on cpu", value=False)
        switch_vae = gr.Checkbox(label="vae on cpu", value=False)
        switch_ue = gr.Checkbox(label="unet_encoder on cpu", value=False)

    try_button.click(fn=start_tryon, inputs=[imgs, garm_img, denoise_steps, seed, gamma], outputs=[image_out,masked_img], api_name='tryon')
    show_button.click(fn=show_device)
    clean_button.click(fn=clean)
    switch_u.change(fn=move_model, inputs=[switch_u, gr.Textbox(value='unet', visible=False)], outputs=None)
    switch_te.change(fn=move_model, inputs=[switch_te, gr.Textbox(value='text_encoder', visible=False)], outputs=None)
    switch_ie.change(fn=move_model, inputs=[switch_ie, gr.Textbox(value='image_encoder', visible=False)], outputs=None)
    switch_vae.change(fn=move_model, inputs=[switch_vae, gr.Textbox(value='vae', visible=False)], outputs=None)
    switch_ue.change(fn=move_model, inputs=[switch_ue, gr.Textbox(value='unet_encoder', visible=False)], outputs=None)
    radio.change(fn=change_resolution, inputs=radio)
    use_opt.change(fn=switch_use_opt, inputs=[use_opt])
print(f"time:{time.time()-ch:.2f}")
image_blocks.launch()

