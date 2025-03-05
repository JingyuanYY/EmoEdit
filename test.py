import argparse

from scripts.model import CombinedModel
import os
import pandas as pd
from collections import defaultdict
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor
)
from tqdm import tqdm


def resize_image_to_512(img):
    # 打开图像
    # 获取当前尺寸
    img = img.convert('RGB')
    width, height = img.size

    # 计算新尺寸
    if width < height:
        new_width = 512
        new_height = int((height / width) * 512)
    else:
        new_height = 512
        new_width = int((width / height) * 512)

    # 调整尺寸并保存
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    return resized_img

def main(para_paths, img_paths, emotion, save_dir):
    Q_Former = CombinedModel(76)

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "/mnt/d/model/instruct-pix2pix", requires_safety_checker=False, safety_checker=None,
        torch_dtype=torch.float16).to("cuda")
    processor = CLIPProcessor.from_pretrained("/mnt/d/model/clip-vit-large-patch14")
    vision_tower_model = CLIPModel.from_pretrained("/mnt/d/model/clip-vit-large-patch14").to(pipeline.device)
    pipeline.set_progress_bar_config(disable=True)
    # run inference
    trained_para = torch.load(para_paths)
    Q_Former.load_state_dict(trained_para)
    Q_Former.eval().to(vision_tower_model.device)

    image_name = os.path.basename(img_paths).split('.')[0]
    with Image.open(img_paths) as validation_image:
        validation_image = resize_image_to_512(validation_image)
        save_path = os.path.join(save_dir, f"{image_name}-{emotion}.jpg")
        os.makedirs(save_dir, exist_ok=True)
        data = processor(images=validation_image, text=emotion, return_tensors="pt", padding="max_length")
        outputs = vision_tower_model(pixel_values=data['pixel_values'].to(vision_tower_model.device),
                                        input_ids=data['input_ids'].to(vision_tower_model.device))
        img_embeds, text_embeds = outputs['image_embeds'].unsqueeze(0).to(vision_tower_model.device), outputs[
            'text_embeds'].unsqueeze(0).to(vision_tower_model.device)
        normal_output = Q_Former(text_embeds=text_embeds, img_embeds=img_embeds)
        with torch.autocast("cuda"):
            image = pipeline(prompt_embeds=normal_output, image=validation_image,
                                guidance_scale=7.5,
                                image_guidance_scale=1.5,
                                num_inference_steps=100).images[0]
            image.save(os.path.join(save_path))


if __name__ == "__main__":

    para_path = 'checkpoint/Q-Former.pth'
    picture_path = 'examples/test.jpg'
    emotion = 'amusement'
    save_dir = 'examples/result'
    os.makedirs(save_dir, exist_ok=True)
    main(para_paths=para_path, img_paths=picture_path, emotion=emotion,
         save_dir=save_dir)
