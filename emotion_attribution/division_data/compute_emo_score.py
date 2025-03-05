import os
import torch
from config import Config
from classifier import Classifier
from transformers import CLIPModel, CLIPProcessor
from PIL import ImageFile
import pandas as pd
from tqdm import tqdm
from img_utils import load_img

ImageFile.LOAD_TRUNCATED_IMAGES = True
weight = "./model_19.pth"
device = "cuda"
# output_dir = '/mnt/d/data/unsplash-research'
# output_dir = '/mnt/d/data/MagicBrush_unzip/source'
# output_dir = '/mnt/d/data/SEED-Data-Edit-Part2-3/real_editing/images'
# output_dir = '/mnt/d/bigProject/img/dataset'
# output_dir = '/mnt/d/data/MA5k/origin_images'
# output_dir = r"D:\data\GQA-inpaint\images_once"
# output_dir = r"D:\data\open-images-dataset\images"
output_dir = r"D:\data\unsplash-research-dataset-full-latest\images"

cfg = Config("./project_config.yaml")
# clip_path = cfg.clip_path
clip_path = r"F:\models\clip-vit-large-patch14"
classifier = Classifier(768, 8).to(device)
state = torch.load(weight, map_location=device)
# classifier.load_state_dict(state)
classifier.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
classifier.eval()

CLIPmodel = CLIPModel.from_pretrained(clip_path).to(device)
processor = CLIPProcessor.from_pretrained(clip_path)

import csv

# csv_file = '/mnt/d/bigProject/results/evaluation_unsplash.csv'
# csv_file = '/mnt/d/bigProject/results/evaluation_SEED-Data-Edit-Part2-3.csv'
# csv_file = '/mnt/d/bigProject/results/evaluation_415.csv'
# csv_file = "./evaluation_GQA.csv"
csv_file = "./evaluation_unsplash_full.csv"
resume = False
mode = "w"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    resume = True
    mode = "a"
else:
    df = None
with open(csv_file, mode=mode, newline="") as file:
    writer = csv.writer(file)
    if not resume:
        writer.writerow(["Filename", "Emotion", "Emotion_score"])
    file_list = os.listdir(output_dir)
    for filename in tqdm(file_list, total=len(file_list)):
        if df is not None and filename in df["Filename"].values:
            print(filename, "has been evaluated")
            continue
        try:
            image_path = os.path.join(output_dir, filename)
            image = load_img(image_path)
            image = processor(images=image, return_tensors="pt", padding=True).to(
                device
            )
            clip = CLIPmodel.get_image_features(**image)
            pred = classifier(clip.to(device))
            pred = torch.softmax(pred, dim=1)
            pred_raw = pred.squeeze(0).cpu().detach().numpy().tolist()
            # print(pred)
            pred_emotion_8 = torch.argmax(pred, dim=1, keepdim=True).item()
            emotion_score = pred_raw[pred_emotion_8]
            # print(pred_emotion_8)
            writer.writerow([filename, cfg.emotion_list[pred_emotion_8], emotion_score])
            print(filename, cfg.emotion_list[pred_emotion_8], emotion_score)
        except:
            print(filename, "error!!!")
        # break
