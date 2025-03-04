import base64
import random
import pandas as pd
import requests
import os
import time
import io
import csv
from PIL import Image

# os.environ["http_proxy"] = "http://127.0.0.1:18008/"  # Specify a proxy to solve connection problems
# os.environ["https_proxy"] = "http://127.0.0.1:18008/"  # Specify a proxy to solve connection problems

# OpenAI API Key
api_key = ""

def encode_image(image_path, new_size=(512, 512)):
    # with open(image_path, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode("utf-8")
    with Image.open(image_path) as img:
        img_resized = img.resize(new_size, Image.ANTIALIAS)
        img_byte_arr = io.BytesIO()
        img_resized.save(img_byte_arr, format='PNG')  
        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    return encoded_image

def summary_cluster(centroid_path):
    base64_image = []
    image_list = [
        os.path.join(centroid_path, img_name) for img_name in os.listdir(centroid_path)
    ]
    emotion_name = image_list[0].split('\\')[-1].split('_')[0]
    image_list = random.sample(image_list, 5)

    for single_img in image_list:
        base64_image.append(encode_image(single_img))
    assert len(base64_image) == 5

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        # "model": "gpt-4-1106-vision-preview",
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # "text": "Generate a concise description of the commonality of an image collection, \
                        # focusing on either objects or actions using\
                        # sentence of 3 words. Identify the main element involved, either the object or the person engaged in the action. \
                        # This clear, focused format improves compatibility with image generation models by specifying distinct elements.\
                        # 3 words: <description>\
                        # Main element: <object or person involved in action>",
                        "text":f"Generate a concise description of the commonality of
                        an image collection, focusing on either objects or actions using sentence of 3 words. Identify the main
                        element involved, either the object or the person engaged in the action. This clear, focused format improves compatibility with image generation models
                        by specifying distinct elements.
                        3 words: <description>
                        Main element: <object or person involved in action>",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image[0]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image[1]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image[2]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image[3]}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image[4]}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 100,
        "seed": 2024,
        "temperature": 0,
    }

    # print(headers)
    # print(payload)
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response


def extract_text(txt):
    my_dict = {}

    lines = txt.split('\n')

    for line in lines:
        if line != '':
            key, value = line.split(': ')
            my_dict[key] = value
    return my_dict


def append_dict_as_row(file_path, dict_of_elem, sep=','):
    df = pd.DataFrame([dict_of_elem])
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        df_existing = pd.read_csv(file_path, sep=sep)
        df_final = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_final = df

    df_final.to_csv(file_path, sep=sep, index=False)


def last_state(file_path):
    try:
        df = pd.read_csv(file_path)
        last_row = df.iloc[-1]
        emotion = last_row['emotion']
        centroid = last_row['centroid']
    except:
        emotion, centroid = None, None
    return emotion, centroid


file_path = 'Summary_20240422_v1.csv'
process = last_state(file_path)
flag = True
for emotion in ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]:
    if (process[0] is not None) & (emotion != process[0]) & flag:
        continue
    else:
        flag = False
    # emotion = "amusement"
    cluster_dir = r"..\example\20240415"
    centroid_dir = os.path.join(cluster_dir, emotion)

    centroid_paths = [os.path.join(centroid_dir, name) for name in os.listdir(centroid_dir)]

    # centroid_paths = sorted(centroid_paths)
    centroid_paths.sort(key=lambda x: int(x.split("_")[-1]))
    results = []
    for index in range(len(centroid_paths))[:]:
        if (process[0] is not None) & (emotion == process[0]):
            if index <= process[1]:
                continue
        print(centroid_paths[index])
        response = summary_cluster(centroid_paths[index])
        content = response.json()["choices"][0]["message"]["content"]
        try:
            tmp_dic = extract_text(content)
            print(content)
        except:
            save_doc = f"Summary_20240422_{emotion}_{index}.txt"
            with open('save_doc', 'w', encoding='utf-8') as file:
                file.write(content)
            print("something wrong")
            exit()
        tmp_dic['emotion'] = emotion
        tmp_dic['centroid'] = index
        # results.append(
        #     {centroid_paths[index]: response.json()["choices"][0]["message"]["content"]}
        # )
        # print(response.json()["choices"][0]["message"]["content"])
        append_dict_as_row(file_path, tmp_dic)
        if index != len(centroid_paths) - 1:
            time.sleep(0.02)
