import plotly.express as px
import pandas as pd
import numpy as np
import utils
from PIL import Image
import os
from collections import defaultdict
from tqdm import tqdm
import base64
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
import shutil
import torch
import json
import torch.nn as nn

class Emotion_cls():
    def __init__(self, weight, device):
        self.device = device
        self.CLIPmodel = CLIPModel.from_pretrained("/mnt/d/model/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("/mnt/d/model/clip-vit-base-patch32")
        self.emotion_list_8 = {"amusement": 0,
                               "awe": 1,
                               "contentment": 2,
                               "excitement": 3,
                               "anger": 4,
                               "disgust": 5,
                               "fear": 6,
                               "sadness": 7}
        self.emotion_list_2 = {"amusement": 0,
                               "awe": 0,
                               "contentment": 0,
                               "excitement": 0,
                               "anger": 1,
                               "disgust": 1,
                               "fear": 1,
                               "sadness": 1
                               }
        self.Emotion8 = ["amusement", "awe", "contentment", "excitement",
                         "anger", "disgust", "fear", "sadness"]
        self.Emotion2 = ["positive", "positive", "positive", "positive",
                         "negative", "negative", "negative", "negative"]
        self.classifier = clip_classifier(8).to(device)
        state = torch.load(weight, map_location=device)
        self.classifier.load_state_dict(state)
        self.classifier.eval()

    def predict_single_img(self, img):  # img must be PIL
        img = img.convert('RGB')
        data = self.processor(images=img, return_tensors="pt", padding=True).to(self.device)
        clip = self.CLIPmodel.get_image_features(**data)
        pred = self.classifier(clip.to(self.device))
        pred_8_emotion = self.Emotion8[torch.argmax(pred).item()]
        pred_2_emotion = self.Emotion2[torch.argmax(pred).item()]
        return pred, pred_8_emotion, pred_2_emotion


class clip_classifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # self.fc = nn.Linear(512, num_classes)
        self.hidden = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

def resize_image_short_side(img, target_size=500):
    original_width, original_height = img.size

    if original_width < original_height:
        new_width = target_size
        new_height = int(original_height * (target_size / original_width))
    else:
        new_height = target_size
        new_width = int(original_width * (target_size / original_height))
    resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_img

def count_clip_txt(summary, tar_img, processor, clip_model):
    with torch.no_grad():
        output = processor(images=tar_img, text=summary, return_tensors="pt", padding="max_length")
        outputs = clip_model(pixel_values=output['pixel_values'].to(clip_model.device),
                            input_ids = output['input_ids'].to(clip_model.device))
        clip_txt = F.cosine_similarity(outputs['image_embeds'], outputs['text_embeds'])
    return clip_txt.item()



def count_clip_dir(summary, origin_img, edited_img, processor, clip_model):
    with torch.no_grad():
        output = processor(images=[origin_img, edited_img], text=summary, return_tensors="pt", padding="max_length")
        outputs = clip_model(pixel_values=output['pixel_values'].to(clip_model.device),
                            input_ids = output['input_ids'].to(clip_model.device))
        dis_img = outputs['img_embeds'][0] - outputs['img_embeds'][1]
        clip_dir = F.cosine_similarity(dis_img.unsqueeze(0), outputs['text_embeds'])
    return clip_dir.item()


def count_aesthetic_score(edited_img, processor, clip_model, aesthetic_model):
    with torch.no_grad():
        output = processor(images=edited_img, return_tensors="pt", padding="max_length")
        outputs = clip_model.get_image_features(**output.to(clip_model.device))
        dis_img = outputs
        score = aesthetic_model(dis_img.to())
    return score.item()


def copy_file_to_directory(src_file, dest_directory):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    shutil.copy(src_file, dest_directory)


def load_text_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def encode_image_base64(image_path):
    with open(image_path, 'rb') as img_file:
        img_bytes = img_file.read()
    return base64.b64encode(img_bytes).decode('utf-8')


def count_CLIP_I(test_img, origin_img, model, processor):
    # torch.cuda.empty_cache()

    data_pro = processor(images=[test_img, origin_img], return_tensors="pt", padding=True).to(model.device)
    data_pro = model.get_image_features(**data_pro)
    d = F.cosine_similarity(data_pro[0, :].unsqueeze(0), data_pro[1, :].unsqueeze(0))
    return d.item()

threshold = (0.75, 0.95)
device = 'cuda'
# emotion classifier
weight = "/mnt/d/code/EmoEdit/emotion_cls/weight/2024-02-23-best.pth"
classifier = Emotion_cls(weight, device)

model = CLIPModel.from_pretrained("/mnt/d/model/clip-vit-base-patch32").to('cuda')
processor = CLIPProcessor.from_pretrained("/mnt/d/model/clip-vit-base-patch32")

aesthetic_model = torch.nn.Linear(512, 1)
s = torch.load("/mnt/d/code/EmoEdit/CVPR/sa_0_4_vit_b_32_linear.pth")
aesthetic_model.load_state_dict(s)
aesthetic_model.eval().to(device)


emotion = ["fear", "amusement", "awe", "anger", "contentment", "excitement", "sadness", "disgust"]
for emo in emotion:
    target_image_dir = f"/mnt/d/data/EmoEdit/test_MagicBrush/{emo}"   #TODO
    target_image_paths = utils.get_all_paths(target_image_dir, ['png', 'jpg'])

    origin_image_dir = "/mnt/d/code/EmoEdit/dataset/test_MagicBrush"   #TODO

    increase_score_list = []
    ssim_list = []
    image_name_list = []
    clip_list = []
    clip_txt_list = []
    img_dic = defaultdict(list)
    for image_path in tqdm(target_image_paths):
        emotion = image_path.split('/')[-2]
        image_name = image_path.split('/')[-1].split('.')[0].strip('_').split('_')[0]
        summary = image_path.split('/')[-1].split('.')[0].split('_')[-1]
        origin_image = os.path.join(origin_image_dir, f"{image_name}_1_source.png")
        if os.path.exists(origin_image) is False:
            origin_image = os.path.join(origin_image_dir, f"__{image_name}.png")
        with Image.open(image_path).convert("RGB") as tar_img:
            with Image.open(origin_image).convert("RGB") as origin_img:
                origin_img = resize_image_short_side(origin_img)
                tar_pred = classifier.predict_single_img(tar_img)[0].squeeze(0)
                origin_pred = classifier.predict_single_img(origin_img)[0].squeeze(0)
                index = classifier.emotion_list_8[emotion]
                increase_score = (tar_pred[index] - origin_pred[index]).item()
                # clip_dir = count_clip_dir(summary=summary, origin_img=origin_img, edited_img=tar_img, processor=processor, clip_model=model)
                clip_txt = count_clip_txt(summary=summary, tar_img=tar_img, processor=processor, clip_model=model)
                clip_txt_list.append(clip_txt)
                aes_score = count_aesthetic_score(edited_img=tar_img, processor=processor, clip_model=model, aesthetic_model=aesthetic_model)
                # ssim = utils.count_ssim(tar_img, origin_img)
                increase_score_list.append(increase_score)
                # ssim_list.append(ssim)
                image_name_list.append(image_path)
                current_clip_i = count_CLIP_I(tar_img, origin_img, model, processor)
                clip_list.append(current_clip_i)
        if (threshold[0] <= current_clip_i <= threshold[1]) and (tar_pred[index] > 0.5) and (clip_txt > 0.25):
            img_dic[image_name].append((image_path, emotion, aes_score))
        

    for img_name, data in img_dic.items():
        best_pair = max(data, key=lambda x: x[2])
        max_path = best_pair[0]
        emotion = best_pair[1]
        Q_dir = f"/mnt/d/data/EmoEdit/test_MagicBrush_cliptxt/{emotion}"
        os.makedirs(Q_dir, exist_ok=True)
        copy_file_to_directory(max_path, Q_dir)
