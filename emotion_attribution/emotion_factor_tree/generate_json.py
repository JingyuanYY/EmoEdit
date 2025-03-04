import json
import pandas as pd
import os
from difflib import SequenceMatcher

image_dir = r"F:\exp\20240409_merge_fix_089_filter_089_big5"


emotion_list = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
]


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def duplicate(data_list):
    # print(data_list)
    seen_names = set()
    new_list = []
    for item in data_list:
        name = item["name"]
        if not any(similar(name, seen_name) > 0.9 for seen_name in seen_names):
            seen_names.add(name)
            new_list.append(item)
        else:
            print(name)
    # print(new_list)
    return new_list


def min_max_normalize(value, min_val, max_val, new_min, new_max):
    return ((value - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min


def count_images_in_folders(image_dir):
    image_counts = {}
    for emotion in emotion_list:
        # Build emotion folder path
        emotion_dir = os.path.join(image_dir, emotion)
        # Build centroid folder path
        centroid_dirs = os.listdir(emotion_dir)
        for centroid_id in centroid_dirs:
            centroid_dir = os.path.join(emotion_dir, centroid_id)
            images = os.listdir(centroid_dir)
            image_count = len(images)
            # Store the count in a dictionary
            image_counts.setdefault(emotion, {})
            image_counts[emotion][centroid_id] = image_count
    return image_counts


image_dir = r"F:\exp\20240409_merge_fix_089_filter_089_big5"
image_counts = count_images_in_folders(image_dir)


emotion = "contentment"
df = pd.read_csv("./Summary_v3_20240513.csv")

df = df[df["emotion"] == emotion]
json_data = {}
json_data["name"] = emotion.title()
json_data["children"] = []

object_data = {"name": "Object", "children": []}
scene_data = {"name": "Scene", "children": []}
action_data = {"name": "Action", "children": []}
facial_data = {"name": "Facial Expression", "children": []}
object_cnt = 0
scene_cnt = 0
action_cnt = 0
facial_cnt = 0

for index, row in df.iterrows():
    # print(row)
    words = row["3 words (v1)"]
    types = row["type(v1)"]
    emotion = row["emotion"]
    centroid_id = row["centroid"]
    value = image_counts[emotion][f"centroid_{centroid_id}"]
    if types == "Object":
        object_data["children"].append({"name": words, "value": value})
        object_cnt += value
    elif types == "Scene":
        scene_data["children"].append({"name": words, "value": value})
        scene_cnt += value
    elif types == "Action":
        action_data["children"].append({"name": words, "value": value})
        action_cnt += value
    elif types == "Facial Expression":
        facial_data["children"].append({"name": words, "value": value})
        facial_cnt += value
    else:
        print(row)

object_data["children"] = duplicate(object_data["children"])
scene_data["children"] = duplicate(scene_data["children"])
action_data["children"] = duplicate(action_data["children"])
facial_data["children"] = duplicate(facial_data["children"])

json_data["children"].append(object_data)
json_data["children"].append(scene_data)
json_data["children"].append(action_data)
json_data["children"].append(facial_data)

object_data["value"] = object_cnt
scene_data["value"] = scene_cnt
action_data["value"] = action_cnt
facial_data["value"] = facial_cnt
json_data["value"] = object_cnt + scene_cnt + action_cnt + facial_cnt

with open(f"./files/record_{emotion}.json", "w") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)
    print("Loading file completed...")
