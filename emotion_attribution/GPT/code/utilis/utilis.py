import os

import pandas as pd
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

def object_color_list(file_path, emotion):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # print(df['emo'])
    # 筛选Emotion列为'amusement'的行
    filtered_df = df[df['emotion'] == emotion]

    # 获取这些行的object列的内容
    objects = filtered_df['3 words (v1)'].tolist()
    colors = filtered_df['Color/lighting (v2)'].tolist()
    # 将objects中的每个元素格式化为"1.XXX"的形式，并组合成一个字符串
    # 使用enumerate来获取每个元素的索引，从1开始编号
    formatted_string = "{" + "; ".join(f"{i}.{obj.strip(' ')} - {colors[i]}" for i, obj in enumerate(objects)) + "}"
    return formatted_string

def object_list(file_path, emotion):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # print(df['emo'])
    # 筛选Emotion列为'amusement'的行
    filtered_df = df[df['emotion'] == emotion]

    # 获取这些行的object列的内容
    objects = filtered_df['3 words (v1)'].tolist()
    types = filtered_df['type(v1)'].tolist()
    # 将objects中的每个元素格式化为"1.XXX"的形式，并组合成一个字符串
    # 使用enumerate来获取每个元素的索引，从1开始编号
    formatted_string = "{" + "; ".join(f"{i}.{types[i]} - {obj}" for i, obj in enumerate(objects)) + "}"
    return formatted_string


def object_color_type_list(file_path, emotion):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # print(df['emo'])
    # 筛选Emotion列为'amusement'的行
    filtered_df = df[df['emotion'] == emotion]

    # 获取这些行的object列的内容
    objects = filtered_df['3 words (v1)'].tolist()
    colors = filtered_df['Color/lighting (v2)'].tolist()
    types = filtered_df['type(v1)'].tolist()
    # 将objects中的每个元素格式化为"1.XXX"的形式，并组合成一个字符串
    # 使用enumerate来获取每个元素的索引，从1开始编号
    formatted_string = "{" + "; ".join(f"{i}.{types[i]} - {obj.strip(' ')} - {colors[i]}" for i, obj in enumerate(objects)) + "}"
    return formatted_string


def extract_text(txt):
    # print(txt)
    lines = txt.split('\n')
    key, value = [], []
    # 遍历每行文本，提取键值对并添加到字典中
    for line in lines:
        if ": " in line:
            object, pos = line.split(': ')
            object = object.strip()
            pos = pos.strip()
            key.append(object)
            value.append(pos)
    return key, value


def append_dict_as_row(file_path, dict_of_elem, sep=','):
    # 将字典转换为DataFrame
    df = pd.DataFrame([dict_of_elem])

    # 检查文件是否存在以及是否为空
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        # 读取现有文件
        df_existing = pd.read_csv(file_path, sep=sep)
        # print("Existing DataFrame:")
        # print(df_existing)
        # 在现有DataFrame上追加新数据
        df_final = pd.concat([df_existing, df], ignore_index=True)
    else:
        # 文件不存在或为空，直接使用新数据
        df_final = df
        # print("New DataFrame:")
        # print(df_final)

    # 将更新后的DataFrame写入CSV文件，不保留索引
    df_final.to_csv(file_path, sep=sep, index=False)


def last_state(file_path):
    try:
        df = pd.read_csv(file_path)

        # 获取最后一行数据
        last_row = df.iloc[-1]

        # 打印最后一行数据及其列标签
        # print("最后一行数据:")
        # print(last_row)

        emotion = last_row['emotion']
        centroid = last_row['No']
    except:
        emotion, centroid = None, None
    return emotion, centroid


def crop_image(image_path, output_size=(512, 512)):
    # with open(image_path, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode("utf-8")
    with Image.open(image_path) as img:
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


def count_ssim(img_1, img_2):
    img_1 = np.array(img_1)
    img_2 = np.array(img_2)
    return ssim(img_1,img_2,channel_axis=2)


