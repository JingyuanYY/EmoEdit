import argparse
import os

import yaml


def get_config(config_path):
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as config:
        cfg = yaml.safe_load(config)
        return cfg


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = self.get_config(config_path)

        self.data_root = self.cfg['EmoSet_path']
        self.emotion_list = self.cfg['emotion_list']
        self.models_path = self.cfg['models_path']
        self.features_path = self.cfg['features_path']
        self.clip_path = self.cfg['clip_path']
        self.summary_path = self.cfg['summary_path']
        self.reason_path = self.cfg['reason_path']

    def get_config(self, config_path):
        if not os.path.exists(config_path):
            return None
        with open(config_path, "r") as config:
            cfg = yaml.safe_load(config)
            return cfg

    def get_config_path(self):
        return self.config_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", type=str, default=None, )
    parser.add_argument("--epoch", type=str, default=None, )
    parser.add_argument("--text_guidance", type=float, default=7.5, )
    parser.add_argument("--image_guidance", type=float, default=1.5, )
    parser.add_argument("--num_inference_steps", type=int, default=30, )
    parser.add_argument("--config_path", type=str, default="/mnt/d/bigProject/config/project_config.yaml", )
    parser.add_argument("--reason_path", type=str, default="/mnt/d/bigProject/data/csv/Reason.csv", )
    parser.add_argument("--projection_type", type=str, default="mlp", )
    parser.add_argument("--prompt_type", type=str, default="instruction", )
    parser.add_argument("--checkpoint_path", type=str, default=None, )

    args = parser.parse_args()
    return args
