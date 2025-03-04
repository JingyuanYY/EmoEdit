from torch.utils.data import Dataset
import utils
from PIL import Image
import torch
import os
from utils import preprocess_images

EMOTION2ID = {"amusement": 0,
              "awe": 1,
              "contentment": 2,
              "excitement": 3,
              "anger": 4,
              "disgust": 5,
              "fear": 6,
              "sadness": 7}


class EmoEditDataset_MultiData(Dataset):
    def __init__(
            self,
            origin_data_root,
            edited_data_root,
            processor,
            vision_tower_model,
            instruction_file_path,
            mixed_precision,
            size=256,
            repeats=1,
            set="train",
            center_crop=False,
    ):
        precision_mapping = {
            'fp16': torch.float16,
            'fp32': torch.float32,
            'bf16': torch.bfloat16
        }
        self.mixed_precision = precision_mapping[mixed_precision]

        self.origin_data_root = origin_data_root
        self.edited_data_root = edited_data_root

        self.edited_image_paths = utils.get_all_paths(self.edited_data_root, ['jpg', 'png'])
        self.targets = [EMOTION2ID[path.split('/')[-2]] for path in self.edited_image_paths]
        self.instruction_dic, self.instruction_list = utils.read_instruction(instruction_file_path)
        self.processor = processor
        self.vision_tower_model = vision_tower_model.to(self.mixed_precision)
        self.size = size

        self.center_crop = center_crop

        self.num_images = len(self.edited_image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        path = self.edited_image_paths[i % self.num_images]
        edited_image = Image.open(path)
        dataset_name = '_'.join(path.split('/')[-3].split('_')[:-1])
        # origin_data = self.processor(images=origin_image, return_tensors="pt", padding=True)

        # example["origin_image"] = origin_data['pixel_values']
        example["name"] = '_'.join(path.split('/')[-1].split('_')[:-1])
        example['summary'] = path.split('/')[-1].split('.')[0].split('_')[-1]
        example["emotion"] = path.split('/')[-2]
        # origin_image = Image.open(os.path.join(self.origin_data_root, f"{example['name']}.png"))
        flag = False
        for dir_name in ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', ]:
            for suffix in ['jpg', 'png']:
                origin_image_path = \
                    os.path.join(self.origin_data_root, f"{dataset_name}_crop/{dir_name}/{example['name']}.{suffix}")
                if os.path.isfile(origin_image_path) is True:
                    origin_image = Image.open(origin_image_path)
                    # origin_image = utils.crop_image(origin_image)
                    flag = True
                    break
            if flag:
                break
        example['instruction'] = 'add ' + example['summary']

        # example['index'] = self.instruction_dic[example["name"]][example["emotion"]][example['summary']][1]

        origin_data = self.processor(images=origin_image, text=[example['emotion']], return_tensors="pt",
                                     padding="max_length")
        edit_data = self.processor(images=edited_image, text=[example['instruction']], return_tensors="pt",
                                   padding="max_length")

        example["origin_image"] = preprocess_images(origin_image, self.size).squeeze(0)
        example["edited_image"] = preprocess_images(edited_image, self.size).squeeze(0)
        example['input_ids'] = origin_data['input_ids']
        with torch.no_grad():
            origin_outputs = self.vision_tower_model(
                pixel_values=origin_data['pixel_values'].to(self.mixed_precision).to(self.vision_tower_model.device),
                input_ids=origin_data['input_ids'].to(self.vision_tower_model.device))
            edit_outputs = self.vision_tower_model(
                pixel_values=edit_data['pixel_values'].to(self.mixed_precision).to(self.vision_tower_model.device),
                input_ids=edit_data['input_ids'].to(self.vision_tower_model.device))
        example['text_embeds'] = origin_outputs['text_embeds']
        example['img_embeds'] = origin_outputs['image_embeds']
        example['instruction_hidden_state'] = edit_outputs['text_model_output']['last_hidden_state'].squeeze(0)
        example['instruction_embeds'] = edit_outputs['text_embeds'].squeeze(0)
        return example
