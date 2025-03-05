import os
import shutil

# # 定义文件夹路径
# folder_path = '/mnt/d/data/MagicBrush_unzip'  # 你需要调整到实际路径
#
# # 定义目标文件夹的名称
# categories = ['source', 'mask', 'target']
#
# # 为每个类别创建文件夹
# for category in categories:
#     os.makedirs(os.path.join(folder_path, category), exist_ok=True)
#
# # 遍历文件夹下的所有文件
# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
#
#     # 检查文件是否是图片文件
#     if os.path.isfile(file_path) and (
#             filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
#
#         # 根据尾缀将文件移动到相应的文件夹
#         for category in categories:
#             if filename.endswith(f'_{category}.png') or filename.endswith(f'_{category}.jpg') or filename.endswith(
#                     f'_{category}.jpeg'):
#                 target_folder = os.path.join(folder_path, category)
#                 shutil.move(file_path, target_folder)
#                 print(f'Moved {filename} to {target_folder}')
# print("文件分类完成")

# folder_path = '/mnt/d/data/MagicBrush_unzip/source'  # 你需要调整到实际路径
# cnt = 0
# for filename in os.listdir(folder_path):
#     if filename.split('_')[1] == '1':
#         cnt += 1
# print(cnt)

import pandas as pd
from tqdm import tqdm

# df = pd.read_csv(r"D:\data/evaluation_GQA.csv")
# base_dir = r"D:\data\GQA-inpaint\images_once"
# output_dir = r"D:\data\GQA-inpaint\images_division"

df = pd.read_csv(r"D:\data\evaluation_unsplash_full.csv")
base_dir = r"D:\data\unsplash-research-dataset-full-latest\images"
output_dir_ori = r"D:\data\unsplash-research-dataset-full-latest\images_division"
output_dir = r"D:\data\unsplash-research-dataset-full-latest\images_division_new"

# 创建文件夹并移动文件
for _, row in tqdm(df.iterrows(), total=len(df)):
    # print(_, row)
    filename = row["Filename"]
    score = row["Emotion_score"]
    # print(filename, score)
    # if filename.split('_')[1] != '1':
    #     continue

    # 计算区间，确保浮点数不出现精度问题
    lower_bound = (score // 0.1) * 0.1  # 保持两位小数
    upper_bound = lower_bound + 0.1

    # 使用字符串格式化，确保输出的是精确的两位小数
    folder_name = f"{lower_bound:.1f}-{upper_bound:.1f}"

    # 创建文件夹
    folder_path = os.path.join(output_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 移动文件到对应的文件夹
    source_file = os.path.join(base_dir, filename)
    destination_file = os.path.join(folder_path, filename)

    destination_file_ori = os.path.join(output_dir_ori, folder_name, filename)

    if os.path.exists(source_file):
        if os.path.exists(destination_file_ori):
            print(f"File {filename} already exists in {folder_name}.")
            continue
        shutil.copy(source_file, destination_file)
        print(f"Moved {filename} to {folder_name}")
    else:
        print(f"File {filename} does not exist.")

import os

# output_dir = "/mnt/d/data/unsplash-research_division"
# output_dir = "/mnt/d/data/MagicBrush_unzip_division"
# output_dir = "/mnt/d/data/SEED-Data-Edit-Part2-3/real_editing/images_division"
cnt = 0
dir_list = os.listdir(output_dir)
dir_list = sorted(dir_list)
for dir_name in dir_list:
    num = os.listdir(os.path.join(output_dir, dir_name))
    cnt += len(num)
    print(dir_name, len(num))
print(cnt)
