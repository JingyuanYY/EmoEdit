import argparse
import os
import torch
import pandas as pd
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from itertools import combinations
import time
import random

EMOTION_LIST = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
]

FEATURE_DIR = {
    "amusement": "features/amusement_clip.pt",
    "anger": "features/anger_clip.pt",
    "awe": "features/awe_clip.pt",
    "contentment": "features/contentment_clip.pt",
    "disgust": "features/disgust_clip.pt",
    "excitement": "features/excitement_clip.pt",
    "fear": "features/fear_clip.pt",
    "sadness": "features/sadness_clip.pt",
}


class UnionFindSet(object):
    def __init__(self, data_list):
        self.father_dict = {}
        self.size_dict = {}

        for node in data_list:
            self.father_dict[node] = node
            self.size_dict[node] = 1

    def find(self, node):
        father = self.father_dict[node]
        if node != father:
            if (
                father != self.father_dict[father]
            ):  # Ensure parent node size dictionary is correct when optimizing for lower tree height
                self.size_dict[father] -= 1
            father = self.find(father)
        self.father_dict[node] = father
        return father

    def is_same_set(self, node_a, node_b):
        return self.find(node_a) == self.find(node_b)

    def union(self, node_a, node_b):
        if node_a is None or node_b is None:
            return

        a_head = self.find(node_a)
        b_head = self.find(node_b)

        if a_head != b_head:
            a_set_size = self.size_dict[a_head]
            b_set_size = self.size_dict[b_head]
            if a_set_size >= b_set_size:
                self.father_dict[b_head] = a_head
                self.size_dict[a_head] = a_set_size + b_set_size
            else:
                self.father_dict[a_head] = b_head
                self.size_dict[b_head] = a_set_size + b_set_size


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# Function to calculate the average similarity within a class
def calculate_average_similarity(
    features, res_dict, centroid_filename, average_similarity, new_rootNode
):
    for old_node in res_dict[new_rootNode]:
        removed_value = average_similarity.pop(old_node, "没有该键(key)")
    node_list = res_dict[new_rootNode]
    centroid_feature = []
    for node_id in node_list:
        centroid_feature += [features[name] for name in centroid_filename[node_id]]
    centroid_feature = torch.cat(centroid_feature)

    features_matrix = np.array(centroid_feature)
    features_matrix_T = features_matrix.T
    centroid_similarity_matrix = np.dot(features_matrix, features_matrix_T)
    diagonal = np.diag(centroid_similarity_matrix)
    centroid_similarity_matrix /= np.sqrt(np.outer(diagonal, diagonal))

    upper_triangle = np.triu(centroid_similarity_matrix, k=1)
    total_similarity = np.sum(upper_triangle)
    num_elements = np.count_nonzero(upper_triangle)

    average_similarity[new_rootNode] = total_similarity / num_elements

    return average_similarity


def compute_similarity_matrix(features, centroid_filename, res_dict):
    emo_centroid_feature = []
    for rootNode in tqdm(res_dict, leave=False):
        node_list = res_dict[rootNode]
        centroid_feature = []
        for node_id in node_list:
            centroid_feature += [features[name] for name in centroid_filename[node_id]]
        centroid_feature = torch.cat(centroid_feature)
        centroid_feature = centroid_feature.mean(dim=0).unsqueeze(0)
        emo_centroid_feature.append(centroid_feature)
    emo_centroid_feature = torch.cat(emo_centroid_feature).detach().numpy()
    features_matrix = np.array(emo_centroid_feature)
    features_matrix_T = features_matrix.T
    similarity_matrix = np.dot(features_matrix, features_matrix_T)
    diagonal = np.diag(similarity_matrix)
    similarity_matrix /= np.sqrt(np.outer(diagonal, diagonal))
    return similarity_matrix, list(res_dict.keys())


def update_similarity_matrix(
    features, similarity_matrix, res_dict, centroid_filename, new_rootNode
):
    for old_node in res_dict[new_rootNode]:
        similarity_matrix[old_node, :] = similarity_matrix[:, old_node] = 0

    new_cluster_features = []
    for node_id in res_dict[new_rootNode]:
        new_cluster_features += [features[name] for name in centroid_filename[node_id]]
    new_cluster_features = torch.cat(new_cluster_features)
    new_cluster_features = new_cluster_features.mean(dim=0)
    new_cluster_features = new_cluster_features.detach().numpy()

    for i in list(res_dict.keys()):
        temp_features = []
        node_list = res_dict[i]
        for node_id in node_list:
            temp_features += [features[name] for name in centroid_filename[node_id]]
        temp_features = torch.cat(temp_features)
        temp_features = temp_features.mean(dim=0)
        temp_features = temp_features.detach().numpy()
        cos_sim = new_cluster_features.dot(temp_features) / (
            np.linalg.norm(new_cluster_features) * np.linalg.norm(temp_features)
        )
        similarity_matrix[new_rootNode, i] = similarity_matrix[i, new_rootNode] = (
            cos_sim
        )

    return similarity_matrix


def main(args):
    random.seed(2333)
    for emotion in EMOTION_LIST[:1]:
        image_list = os.listdir(os.path.join(args.image_dir, emotion))[:]
        image_list = [name.split(".")[0] for name in image_list]
        image_list = sorted(image_list, key=lambda x: int(x.split("_")[-1]))
        image_num = len(image_list)
        node_list = range(len(image_list))

        centroid_filename = {}
        for index in range(len(image_list)):
            centroid_filename[index] = [image_list[index]]
        print(f"{emotion} have {image_num} files!!!")
        union_find_set = UnionFindSet(node_list)
        features = torch.load(FEATURE_DIR[emotion]) # Load features

        res_dict = {num: [num] for num in range(len(centroid_filename))}

        average_similarity = {}

        dynamic_threshold = args.threshold  # Initialize threshold

        cluster_bar = tqdm(range(image_num), desc=emotion)
        for now_epoch in cluster_bar:
            if now_epoch == 0:
                similarity_matrix, cluster_index = compute_similarity_matrix(
                    features, centroid_filename, res_dict
                )
            np.fill_diagonal(similarity_matrix, 0)
            max_index = np.argmax(similarity_matrix)
            max_i, max_j = np.unravel_index(max_index, similarity_matrix.shape)
            max_value = similarity_matrix[max_i][max_j]
            if max_value < dynamic_threshold:
                print(f"No higher than {dynamic_threshold}. ---- {now_epoch}")
                break
            union_find_set.union(cluster_index[max_i], cluster_index[max_j])
            res_dict = {}
            for i in union_find_set.father_dict:
                rootNode = union_find_set.find(i)
                if rootNode in res_dict:
                    res_dict[rootNode].append(i)
                else:
                    res_dict[rootNode] = [i]
            new_cluster = union_find_set.find(cluster_index[max_i])
            average_similarity = calculate_average_similarity(
                features, res_dict, centroid_filename, average_similarity, new_cluster
            )
            update_similarity_matrix(
                features, similarity_matrix, res_dict, centroid_filename, new_cluster
            )
            cluster_average_similarity = np.mean(list(average_similarity.values()))
            dynamic_threshold = min(cluster_average_similarity, args.threshold)
            if dynamic_threshold < args.threshold:
                print(f"Threshold: {dynamic_threshold} < {args.threshold}   !!!")
                break
            cluster_bar.set_postfix(
                cluster_average_similarity=cluster_average_similarity,
                dynamic_threshold=dynamic_threshold,
            )

        # print(res_dict)
        cnt = 0
        for rootNode in res_dict:
            node_list = res_dict[rootNode]
            new_node_dir = os.path.join(args.output_dir, emotion, f"centroid_{cnt}")
            if len(node_list) <= 2:
                continue
            if not os.path.exists(new_node_dir):
                os.makedirs(new_node_dir, exist_ok=True)
            for node_id in node_list:
                filename = centroid_filename[node_id][0]
                old_path = os.path.join(args.image_dir, emotion, f"{filename}.jpg")
                new_path = os.path.join(
                    args.output_dir,
                    emotion,
                    f"centroid_{cnt}",
                    f"{filename}.jpg",
                )
                shutil.copy(old_path, new_path)
            cnt += 1

        print("Number of clusters:", len(res_dict.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/mnt/sdb1/jingyuan/data/EmoSet_v5/image", #TODO
    )
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )
    parser.add_argument("--seed", default=2333, type=int, help="just a random seed")
    parser.add_argument(
        "--output_dir", default="results/20240407_merge_adaptive_mean_085" #TODO
    )
    parser.add_argument("--threshold", default=0.85, type=float)
    opt = parser.parse_args()

    main(opt)
