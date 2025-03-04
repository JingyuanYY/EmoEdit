import argparse

from model import CombinedModel
import os
import pandas as pd
from collections import defaultdict
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image
from utils import get_all_paths
from transformers import (
    CLIPModel, CLIPProcessor
)
from tqdm import tqdm


# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def main(root, para_paths, save_dir, seed):
    Q_Former = CombinedModel(76)

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "/mnt/d/model/instruct-pix2pix", requires_safety_checker=False, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    processor = CLIPProcessor.from_pretrained("/mnt/d/model/clip-vit-large-patch14")
    vision_tower_model = CLIPModel.from_pretrained("/mnt/d/model/clip-vit-large-patch14").to(pipeline.device)
    pipeline.set_progress_bar_config(disable=True)
    # run inference
    generator = torch.Generator(device=vision_tower_model.device).manual_seed(seed)
    num_validation_images = 1
    validation_images_path = get_all_paths(root,['jpg','png'])
    # validation_images_path = validation_images_path[:5]
    # validation_images_path = ["/mnt/d/code/EmoEdit/dataset/dataset_crop/61.png",]
                            #   "/mnt/d/code/EmoEdit/dataset/dataset_crop/180.png",
                            #   "/mnt/d/code/EmoEdit/dataset/dataset_crop/195.png",
                            #   "/mnt/d/code/EmoEdit/dataset/dataset_crop/222.png",
                            #   "/mnt/d/code/EmoEdit/dataset/dataset_crop/224.png",
                            #   "/mnt/d/code/EmoEdit/dataset/dataset_crop/253.png",
                            #   "/mnt/d/code/EmoEdit/dataset/dataset_crop/296.png",
                            #   "/mnt/d/code/EmoEdit/dataset/dataset_crop/362.png",]
    # validation_images = Image.open(validation_images_path)

    emotion_list = ['amusement', 'awe', 'contentment', 'excitement',
                    'disgust', 'anger', 'fear', 'sadness']
    for para_path in tqdm(para_paths):
        step = os.path.basename(para_path).split('.')[0].split('_')[1]
        # print(step)
        trained_para = torch.load(para_path)
        Q_Former.load_state_dict(trained_para)
        Q_Former.eval().to(vision_tower_model.device)
        for path in validation_images_path:
            image_name = os.path.basename(path).split('.')[0]
            with Image.open(path) as validation_image:
                for emotion in emotion_list:
                    generator = torch.Generator(device=vision_tower_model.device).manual_seed(250)

                    save_path = os.path.join(save_dir, f"{step}")
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{step}/{image_name}_{emotion}_step-{step}_seed-{seed}_server-168.jpg")
                    if os.path.isfile(save_path):
                        continue
                    images = []
                    data = processor(images=validation_image, text=emotion, return_tensors="pt", padding="max_length")
                    outputs = vision_tower_model(pixel_values=data['pixel_values'].to(vision_tower_model.device), input_ids=data['input_ids'].to(vision_tower_model.device))
                    img_embeds, text_embeds = outputs['image_embeds'].unsqueeze(0).to(vision_tower_model.device), outputs['text_embeds'].unsqueeze(0).to(vision_tower_model.device)
                    normal_output = Q_Former(text_embeds=text_embeds, img_embeds=img_embeds)

                    for _ in range(num_validation_images):
                        with torch.autocast("cuda"):
                            image = pipeline(prompt_embeds=normal_output, image=validation_image, guidance_scale=7.5,
                                            image_guidance_scale=1.5,
                                            num_inference_steps=30, generator=generator).images[0]
                        images.append(image)
                    for i, img in enumerate(images):
                        img.save(save_path)
                    save_path = os.path.join(train_datadir, f'hidden_state/{step}')
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(normal_output.to('cpu'), os.path.join(train_datadir, f'hidden_state/{step}/{image_name}_{emotion}_step-{step}_seed-{seed}.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, action='append', default=[], help="choose which emotion dir",)
    args = parser.parse_args()
    root = "/mnt/d/code/EmoEdit/dataset/dataset_crop"
    train_datadir = "/mnt/d/code/EmoEdit/CVPR/train_data/10-28_Rate10_LR-4_Newdataset_Ablation_DiffusionLossOnly/"
    seed = 48000
    # para_names = [ i for i in range(1000,8000,1000)]
    # para_names = [40000] 
    para_names = args.step
    para_paths = [os.path.join(train_datadir, f"Q-Former/Q-Former_{str(name)}.pth") for name in para_names]
    save_dir = os.path.join(train_datadir, "validation-on-TrainSet-all-30")
    os.makedirs(save_dir, exist_ok=True)
    main(root=root, para_paths=para_paths, save_dir=save_dir, seed=seed)
