import os
import pandas as pd
from collections import defaultdict
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image
import torch.nn.functional as F
import json
from skimage.metrics import structural_similarity as ssim
import numpy as np


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def convert_strings_to_numbers(string_list):
    string_to_number = {}
    current_number = 0
    result = []

    for s in string_list:
        if s not in string_to_number:
            string_to_number[s] = current_number
            current_number += 1
        result.append(string_to_number[s])

    return result


def preprocess_images(image, resolution):
    original_images = convert_to_np(image, resolution)
    # We need to ensure that the original and the edited images undergo the same
    # augmentation transforms.
    images = torch.tensor(original_images)
    images = 2 * (images / 255) - 1
    final_image = images.reshape(-1, 3, resolution, resolution)
    return final_image



def get_all_paths(dir_root, suffix: list):
    txt_file_list = []
    for root, _, file_path in os.walk(dir_root):
        for file in file_path:
            for suffix_name in suffix:
                if file.endswith(suffix_name):
                    tmp = os.path.join(root, file)
                    txt_file_list.append(tmp)
    txt_file_list.sort()
    return txt_file_list

def save_text_to_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def load_text_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    

def count_ssim(img_1, img_2):
    img_1 = np.array(img_1)
    img_2 = np.array(img_2)
    return ssim(img_1,img_2,channel_axis=2)


def crop_image(img, output_size=(512, 512)):
    # with open(image_path, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode("utf-8")
        # 调整图像到指定大小
    width, height = img.size

    # 计算中心裁剪的尺寸
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2

    # 裁剪图像
    img_cropped = img.crop((left, top, right, bottom))

    # 调整图像大小
    img_resized = img_cropped.resize(output_size)
    # img_resized = img.resize(new_size, Image.ANTIALIAS)

    return img_resized

