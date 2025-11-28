import gradio as gr
import cv2
import os
from PIL import Image
import numpy as np
import torch, torchtext
import random
from transformers import AutoTokenizer, PretrainedConfig
from versagen.versagen_models.unet import UNet
from versagen.versagen_models.controlnet import ControlNet
from utils.gaussian_smoothing import GaussianSmoothing
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from tqdm import tqdm
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
import math
from utils.adjust_boxes import boxes_process
import torch.nn as nn
from time import time
from versagen.attention2mask import Attention2Mask

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

device1 = torch.device('cuda')

temp_dir = './temp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
os.environ['GRADIO_TEMP_DIR'] = temp_dir


MAX_STEP = 50
INJECT_STEP = 2

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
def get_index(sentence, target_word):
    tokens = [token.lower() for token in tokenizer(sentence)]
    target_word = target_word.replace(" ", "").lower()
    
    vocab = torchtext.vocab.build_vocab_from_iterator([tokens], specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    
    word_indices = [vocab[word] for word in tokens]
    
    try:
        target_index = word_indices.index(vocab[target_word])
    except ValueError:
        target_index = -1
    
    return target_index

def setup_seed(seed):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)   
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)
    pass

pretrained_path = "stabilityai/stable-diffusion-2-1"
controlnet_path = "./checkpoint/model_00.pth"

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":  
        from transformers import CLIPTextModel
        return CLIPTextModel
    
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids      

        prompt_embeds = text_encoder(               
            text_input_ids.to(text_encoder.device)
            )[0]

        return prompt_embeds   

def find_max_region(mask):
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    center, radius = cv2.minEnclosingCircle(largest_contour)
    center = tuple(map(int, center))
    radius = int(radius)
    
    max_area = np.zeros_like(mask)
    max_area = cv2.fillPoly(max_area, [largest_contour], 255)

    return max_area, center, radius

smoothing = GaussianSmoothing().to(device1)
def _attn_smoothing(image):
    input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    image = smoothing(input).squeeze(0).squeeze(0)  # (16,16)
    image = image
    return image

with torch.cuda.device("cuda"):
    tokenizer_one = AutoTokenizer.from_pretrained(
            pretrained_path, subfolder="tokenizer", revision=None, use_fast=False, torch_dtype=torch.float16,
        )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_path, None
    )

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder="scheduler", torch_dtype=torch.float16)
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        pretrained_path, subfolder="text_encoder", revision=None, torch_dtype=torch.float16,
    ).to(device1)

    vae = AutoencoderKL.from_pretrained(
        pretrained_path,
        subfolder="vae",
        revision=None,
        torch_dtype=torch.float16,
    ).to(device1)

    unet = UNet.from_pretrained(
        pretrained_path, subfolder="unet", revision=None, torch_dtype=torch.float16,
    ).to(device1)

    controlnet = ControlNet.from_unet(unet).to(device=device1, dtype=torch.float16)
    checkpoint = torch.load(controlnet_path, map_location='cpu')
    controlnet.load_state_dict(checkpoint)

    vae.eval()
    unet.eval()
    controlnet.eval()
    text_encoder_one.eval()
    text_encoders = [text_encoder_one]
    tokenizers = [tokenizer_one]

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    scale_range = (1., 0.5)
    scale_range = np.linspace(scale_range[0], scale_range[1], 3)
    scale_factor = 18.  

is_adjust = True
box_adjust = True

gamma_a = 0.7
gamma_b = 0.6
gamma_c = 15        
max_size = 0.7
min_size = 0.3
alpha = 0.1
count_adjust = 15
iou_threshold=0.005
unet_channels = unet.config.in_channels

a2m = Attention2Mask(unet, image_size=(768, 768))

def process(prompt, word1, word2, word3, sketch1, sketch2, sketch3, seed, guidance_scale, alpha_pixel, alpha_token):
    # with torch.cuda.device(device_num_1):
    start_time = time()
    global unet, controlnet
    
    obj_limit = 3
    img_size = 768

    sketches = [sketch1, sketch2, sketch3]
    words_list = [word1, word2, word3]
    
    index_list = []
    bad_inx = []
    for i in range(obj_limit):
        if words_list[i] == "":
            bad_inx.append(i)
            index_list.append(-1)
        else:
            sk_idx = get_index(prompt, words_list[i])
            index_list.append(sk_idx)
            if sk_idx == -1:
                bad_inx.append(i)
    
    if len(bad_inx) == 3 or prompt=="":
        gr.Warning("输入条件有误，请重新输入")
        error_img = Image.open("./demo/error.png")
        return [error_img, error_img], 0.    
    
    index_list_temp = []
    sketches_temp = []
    words_list_temp = []
    for i in range(obj_limit):
        if i not in bad_inx:
            index_list_temp.append(index_list[i])
            sketches_temp.append(sketches[i])
            words_list_temp.append(words_list[i])
    
    index_list = index_list_temp
    sketches = sketches_temp
    words_list = words_list_temp
    del index_list_temp, sketches_temp, words_list_temp
    
    prompt_text = [prompt]
    negative_prompt = ["anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"]
    
    sample_idx = torch.ones(obj_limit, dtype=torch.int) * -1
    sample_egde = torch.FloatTensor(torch.zeros([obj_limit, 3, img_size, img_size],  dtype=torch.float))
    sample_mask = torch.FloatTensor(torch.zeros([obj_limit, img_size, img_size],  dtype=torch.float))
    
    for i in range(len(index_list)):
        
        sample_idx[i] = index_list[i]
        
        _, obj_edge = cv2.threshold(sketches[i], 127, 255, cv2.THRESH_BINARY)
        obj_edge = 255 - obj_edge
        obj_edge = Image.fromarray(obj_edge)
        
        w, h = obj_edge.size
        scale = max(w, h)/float(img_size)
        new_w, new_h = int(w/scale), int(h/scale)
        obj_edge = obj_edge.resize((new_w, new_h))
        resize_edge = Image.new("L", (img_size, img_size), 0)
        position = ((img_size - new_w) // 2, (img_size - new_h) // 2)
        resize_edge.paste(obj_edge, position)
        sketch = torch.FloatTensor(np.array(resize_edge) / 255.)
        
        sample_egde[i] = sketch

        _, mask = cv2.threshold(sketches[i], 128, 255, cv2.THRESH_BINARY)
        mask = 255 - mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=9)
        mask = cv2.erode(mask, kernel, iterations=9)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        max_contour = contours[0]

        epsilon = 0.001*cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        mask = np.zeros(mask.shape[:2], np.uint8)
        cv2.drawContours(mask, [approx], -1, 255, -1)
        
        kernal = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernal, iterations=5)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        mask = cv2.fillPoly(mask, [largest_contour], 255)

        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask = torch.FloatTensor(mask / 255.).squeeze(0)
        sample_mask[i, position[1]:position[1]+new_h, position[0]:position[0]+new_w] = mask
    
    sample_idx = sample_idx.unsqueeze(0).to(device1)
    sample_egde = sample_egde.unsqueeze(0).to(device1).to(dtype=torch.float16)
    sample_mask = sample_mask.unsqueeze(0).to(device1).to(dtype=torch.float16)
    
    height = img_size
    width = img_size
    setup_seed(seed)
    latent = torch.randn(1, unet_channels, height // 8, width // 8)
    latents = latent.expand(1,  unet_channels, height // 8, width // 8).to(device1)
    latents = torch.cat([latents] * 2).to(dtype=torch.float16)
    noise_scheduler.set_timesteps(MAX_STEP)
    timesteps = noise_scheduler.timesteps
    
    with torch.no_grad():
        prompt_embeds = encode_prompt(prompt_text, text_encoders, tokenizers, 0, is_train=False)
        negative_prompt_embeds = encode_prompt(
            negative_prompt, text_encoders, tokenizers, 0, is_train=False)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_embeds = prompt_embeds.to(device1, dtype=torch.float16)

        count_t = 0
        setup_seed(seed)
        
        for s, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            gamma_t = 1-gamma_a*(1/(1+math.exp(-gamma_b*(count_t-gamma_c))))
            
            if count_t == INJECT_STEP:
                sample_idx = sample_idx.to(device1)
                attns = a2m.get_net_attn_map()
                
                masks = torch.zeros([*sample_idx.shape, 768, 768], dtype=sample_egde.dtype).to(device1)
                mask_all = torch.zeros([768, 768], dtype=sample_egde.dtype).to(device1)
                sample_egde_ori =  torch.zeros([*sample_idx.shape, 768, 768], dtype=sample_egde.dtype).to(device1)
                edge_all = torch.zeros([768, 768], dtype=sample_egde.dtype).to(device1)
                mask_obj_all = torch.zeros([768, 768], dtype=sample_egde.dtype).to(device1)
                coordinations = []
                for j in range(sample_idx.shape[1]):
                    if sample_idx[0, j] == -1:
                        continue
                    
                    attn = attns[0, sample_idx[0, j]+1]
                    
                    min_val = torch.min(attn)
                    max_val = torch.max(attn)
                    attn = ((attn - min_val) / (max_val - min_val))
                    attn = _attn_smoothing(attn)
                    min_val = torch.min(attn)
                    max_val = torch.max(attn)
                    attn = ((attn - min_val) / (max_val - min_val))
                    
                    attn = attn * 255.
                    
                    attn = attn.cpu().numpy().astype(np.uint8)
                    attn = cv2.resize(attn, (768, 768))
                
                    _, mask = cv2.threshold(attn, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    kernal = np.ones((9, 9), np.uint8)
                    mask, center, radius = find_max_region(mask)
                    side_length = int(radius * np.sqrt(2))
                    side_length = side_length + int(0.15 * 768)
                    if side_length <= min_size * 768:
                        side_length = int(min_size * 768)
                    elif side_length > max_size * 768:
                        side_length = int(max_size * 768)
                        
                    square_top_left = (max(0, center[0] - side_length // 2), max(0, center[1] - side_length // 2))
                    square_bottom_right =   (
                                                min(768, square_top_left[0] + side_length),
                                                min(768, square_top_left[1] + side_length)
                                            )
                    
                    x0, y0 = square_top_left
                    x1, y1 = square_bottom_right
                    
                    coordinations.append([x0,y0,x1,y1])
                    
                coordinations = np.array(coordinations)
                if len(coordinations)>1 and box_adjust:
                    coordinations = boxes_process(coordinations, img_size, iou_threshold=iou_threshold, min_size=min_size)
                
                for j in range(sample_idx.shape[1]):
                    if sample_idx[0, j] == -1:
                        continue
                    
                    x0,y0,x1,y1 = coordinations[j]
                    
                    edge = sample_egde[0, j, 0]
                    edge = F.interpolate(
                                            edge.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32),
                                            size=(y1-y0, x1-x0),
                                            mode='nearest'
                                        ).squeeze(0).squeeze(0)

                    sample_egde_ori[0, j, y0:y1, x0:x1] += edge
                    edge_all[y0:y1, x0:x1] = torch.where(edge != 0., edge, edge_all[y0:y1, x0:x1])
                    
                    mask_obj = sample_mask[0, j]
                    mask_obj = F.interpolate(
                                            mask_obj.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32),
                                            size=(y1-y0, x1-x0),
                                            mode='nearest'
                                        ).squeeze(0).squeeze(0)
                    
                    masks[0, j, y0:y1, x0:x1] = torch.where(mask_obj!= 0., mask_obj, masks[0, j, y0:y1, x0:x1])
                    mask_all[y0:y1, x0:x1] = 1.
                    mask_obj_all[y0:y1, x0:x1] = torch.where(mask_obj!= 0., mask_obj, mask_obj_all[y0:y1, x0:x1])
                
                sample_egde_ori = sample_egde_ori.unsqueeze(2).repeat((1, 1, 3, 1, 1)).to(dtype=torch.float16)
                mask_all = (mask_all * 255.).cpu().numpy().astype(np.uint8)
                kernal = np.ones((9, 9), np.uint8)
                mask_all = cv2.dilate(mask_all, kernal, iterations=3)
                mask_all = (torch.tensor(mask_all) / 255.).to(device1).to(dtype=torch.float16)

                if is_adjust:
                    count_adjust = 20
                    adjust_type = torch.float32
                    unet = unet.to(adjust_type)
                    while count_adjust>0 and is_adjust:
                        latent = latents[-1].unsqueeze(0)
                        del latents
                        torch.cuda.empty_cache()
                        prompt_embed = prompt_embeds[-1].unsqueeze(0)
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            latent,
                            t,
                            encoder_hidden_states=prompt_embed,
                            controlnet_cond=sample_egde_ori,
                            return_dict=False,
                            conditioning_scale=gamma_t
                        ) 
                        torch.cuda.empty_cache()
                        
                        with torch.enable_grad():
                            a2m.clean()
                            latent = latent.clone().detach().requires_grad_(True)
                            torch.cuda.empty_cache()
                            _ = unet(
                                        latent.to(adjust_type),
                                        t,
                                        encoder_hidden_states=prompt_embed.to(adjust_type),
                                        down_block_additional_residuals=[
                                            sample.to(adjust_type) for sample in down_block_res_samples
                                        ],
                                        mid_block_additional_residual=mid_block_res_sample.to(adjust_type),
                                        return_dict=False,
                                    )[0]

                            del _, down_block_res_samples, mid_block_res_sample
                            torch.cuda.empty_cache()
                            attns = a2m.get_net_attn_map()[0]
                            a2m.clean()
                            torch.cuda.empty_cache()
                            
                            token_loss = 0.0         
                            j, H, W = attns.shape
                            masks_resize = F.interpolate(
                                        masks,
                                        size=(H, W),
                                        mode='nearest'
                                    )
                            
                            token_loss = 0.0
                            for j in range(sample_idx.shape[1]):
                                if sample_idx[0, j] == -1:
                                    continue
                                
                                mask = masks_resize[0, j]
                                attn_i = attns[sample_idx[0, j]+1]
                                
                                activation_value = (attn_i * mask).reshape(-1).sum(dim=-1)/attn_i.reshape(-1).sum(dim=-1)
                                token_loss = token_loss + (1.0 - activation_value) ** 2
                            
                            token_loss = token_loss/len(index_list)
                            
                            bce_loss_func = nn.BCELoss()
                            pixel_loss = 0.0
                            for j in range(sample_idx.shape[1]):
                                if sample_idx[0, j] == -1:
                                    continue
                                attn_i = attns[sample_idx[0, j]+1].to(adjust_type)
                                
                                
                                
                                H, W = attn_i.shape
                                
                                mask = masks[0, j].to(adjust_type)
                                mask = F.interpolate(
                                            mask.unsqueeze(0).unsqueeze(0),
                                            size=(H, W),
                                            mode='nearest'
                                        ).squeeze(0).squeeze(0)
                                
                                pixel_loss = pixel_loss + bce_loss_func(attn_i.unsqueeze(0), 
                                                            mask.unsqueeze(0))
                                
                            pixel_loss = (pixel_loss.cpu()/len(index_list))

                            del masks_resize
                            a2m.clean()
                            torch.cuda.empty_cache()
                            
                            grounding_loss = pixel_loss * alpha * alpha_pixel + token_loss * (1-alpha)*alpha_token
                            
                            print("grounding_loss: ", grounding_loss.item(), 
                                "pixel_loss: ", pixel_loss.item(), 
                                "token_loss", token_loss.item())
                            
                            if pixel_loss < 0.30:
                                alpha_pixel = 0.
                            if token_loss < 0.4:
                                alpha_token = 0
                            
                            grad_cond = torch.autograd.grad(grounding_loss.requires_grad_(True), 
                                                            [latent], retain_graph=True)[0]
                            
                        step_size = scale_factor * np.sqrt(scale_range[count_t-2])
                        mask_all_resize = F.interpolate(
                                            mask_all.unsqueeze(0).unsqueeze(0),
                                            size=grad_cond.shape[-2:],
                                            mode='nearest'
                                        ).squeeze(0).squeeze(0)

                        latent = latent - step_size * grad_cond.detach() * mask_all_resize
                        latents = torch.cat([latent] * 2)
                        
                        count_adjust -= 1
                        del attns, prompt_embed, attn_i, mask, mask_all_resize, activation_value, grounding_loss
                        del grad_cond, pixel_loss, token_loss
                        a2m.clean()
                        del latent
                        torch.cuda.empty_cache()
                        
                        if alpha_pixel==0 and alpha_token==0:
                            break

                    a2m.clean()
                    unet = unet.to(torch.float16)
                    torch.cuda.empty_cache()
            
            latents = noise_scheduler.scale_model_input(latents, t)
            
            if count_t < INJECT_STEP:
                a2m.clean()
                model_pred = unet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            else : 
                a2m.clean()
                placeholder = torch.cat([sample_egde_ori, sample_egde_ori], dim=0)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=placeholder,
                    return_dict=False,
                    conditioning_scale=gamma_t
                )

                model_pred = unet(
                                latents,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                down_block_additional_residuals=[
                                    sample.to(dtype=torch.float16) for sample in down_block_res_samples
                                ],
                                mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float16),
                                return_dict=False,
                            )[0]

                a2m.clean()
                
            noise_pred_uncond, noise_prediction_local = model_pred.chunk(2)
            noise_pred = noise_pred_uncond +\
                            guidance_scale * (noise_prediction_local - noise_pred_uncond)
            latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            count_t += 1
        
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil")[0]
        edge_all = ((1.-edge_all) * 255.).cpu().numpy().astype(np.uint8)
        edge_all = Image.fromarray(edge_all)
    
    end_time = time()
    duration = "%.2f"%(end_time - start_time)

    del latents, noise_pred_uncond, noise_prediction_local, model_pred, noise_pred
    del index_list, sketches, words_list
    del prompt_embeds, down_block_res_samples, mid_block_res_sample, placeholder
    a2m.clean()
    del  sample_idx, sample_egde, sample_mask
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    return [image, edge_all], duration    

def fun_clear(*args):
    result = []
    for arg in args:
        if isinstance(arg, list):
            result.append([])
        else:
            result.append(None)
    return tuple(result)

custom_css = """
#centered-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
}
"""

with gr.Blocks(title="VersaGen", css=custom_css) as versagen:
    gr.Markdown("<h1 id='centered-title'>VersaGen</h1>")
    with gr.Tab(label="VersaGen"):
        with gr.Row():
            with gr.Column():
                text0 = gr.Textbox(label="Prompt", scale=1, min_width=1, interactive=True)
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            # sketch1 = gr.Sketchpad(label="sketch 1", shape=(768, 768), 
                            #               image_mode="L", brush_radius=20., interactive=True)
                            sketch1 = gr.Image(label="sketch 1", 
                                          image_mode="L", interactive=True)
                            text1 = gr.Textbox(label="name of the sketch 1", scale=1, min_width=1, interactive=True)
                        
                        with gr.Column():
                            # sketch2 = gr.Sketchpad(label="sketch 2", shape=(768, 768), 
                            #               image_mode="L", brush_radius=20., interactive=True)
                            sketch2 = gr.Image(label="sketch 2", 
                                          image_mode="L", interactive=True)
                            text2 = gr.Textbox(label="name of the sketch 2", scale=1, min_width=1, interactive=True)
                    
                    with gr.Row():
                        with gr.Column():
                            # sketch3 = gr.Sketchpad(label="sketch 3", shape=(768, 768), 
                            #               image_mode="L", brush_radius=20., interactive=True)
                            sketch3 = gr.Image(label="sketch 3", 
                                          image_mode="L", interactive=True)
                            text3 = gr.Textbox(label="name of the sketch 3", scale=1, min_width=1, interactive=True)

                        with gr.Column():
                            with gr.Row():
                                seed = gr.Slider(minimum=0, maximum=1000, label="Radom Seed", 
                                                scale=4, min_width=1, value=42, interactive=True)
                                
                            cfg = gr.Slider(minimum=0, maximum=10, label="CFG strength", value=7.5, interactive=True)
                            alpha_pixel = gr.Slider(minimum=0, maximum=50, label="alpha pixel loss", value=10., interactive=True)
                            alpha_token = gr.Slider(minimum=0, maximum=50, label="alpha token loss", value=10., interactive=True)
            
            with gr.Column():
                output = gr.Gallery(label="Output")
                duration = gr.Number(label="Duration", value=0)
                with gr.Row():
                    button1 = gr.Button(value="Generate! ", scale=1, min_width=1, variant="primary")
                    clear_button = gr.Button("Clear", scale=1, min_width=1)
        
        inputs = [
            text0, text1, text2, text3,
            sketch1, sketch2, sketch3,
            seed, cfg, alpha_pixel, alpha_token
        ]
        
        with gr.Column():
            gr.Markdown("## Examples")
            examples = [
                [
                    "The squirrel investigates the pizza slice left on the picnic blanket.",
                    "pizza",
                    "squirrel",
                    None,
                    "versagen_demo/1/pizza_sk.png",
                    "versagen_demo/1/squirrel_sk.png",
                    None,
                    None,
                    None,
                    None,
                    None
                ],
                [
                    "A bear and a deer are in the rainforest.", 
                    "bear", 
                    "deer",
                    None, 
                    "versagen_demo/2/bear_sk.png",
                    "versagen_demo/2/deer_sk.png",
                    None, 
                    None, 
                    None, 
                    None,
                    None 
                ],
                [
                    "A car and an airplane move in front of Mount Fuji.",
                    "airplane", 
                    "car",
                    None,
                    "versagen_demo/3/airplane_sk.png",
                    "versagen_demo/3/car_sk.png",
                    None, 
                    None,
                    None,
                    None,
                    None
                ],
                [
                    "The candle and a bell and a bird on the desk",
                    "bell",
                    "bird",
                    "candle",
                    "versagen_demo/4/bell_sk.png",
                    "versagen_demo/4/bird_sk.png",
                    "versagen_demo/4/candle_sk.png",
                    None,
                    None,
                    None,
                    None
                ],
                [
                    "A car and a giraffe and an airplane are on the road",
                    "airplane",
                    "car",
                    "giraffe",
                    "versagen_demo/5/are_sk.png",
                    "versagen_demo/5/car_sk.png",
                    "versagen_demo/5/giraffe_sk.png",
                    None,
                    None, 
                    None,
                    None 
                ],
            ]
            
            gr.Examples(
                examples=examples,
                inputs=inputs,
            )
        
        button1.click(fn=process, inputs=inputs, outputs=[output, duration])

        clear_button.click(fn=fun_clear, 
                           inputs=[text0, text1, text2, text3, sketch1, sketch2, sketch3, output, duration],
                           outputs=[text0, text1, text2, text3, sketch1, sketch2, sketch3, output, duration])

versagen.queue(max_size=20).launch(server_name='0.0.0.0', share=True, show_error=True, server_port=7899)

