import argparse
import os
import re

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from ACC_args import emo_cls
import lpips
from classifier_model import clip_classifier
from tqdm import tqdm

def crop_img(img, output_size=(512,512)):
    width, height = img.size

    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2

    img_cropped = img.crop((left, top, right, bottom))

    img_resized = img_cropped.resize(output_size)
    return img_resized

def count_ssim(test_img_array, origin_img_array):
    return ssim(test_img_array, origin_img_array, channel_axis=2)


def count_psnr(test_img_array, origin_img_array):
    return psnr(test_img_array, origin_img_array, data_range=origin_img_array.max() - origin_img_array.min())

def count_mse(test_img_array, origin_img_array):
    return np.mean((test_img_array - origin_img_array) ** 2)

def count_lpips(test_img_array, origin_img_array, lpips_alex):

    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])

    in0 = transform(test_img_array)
    in1 = transform(origin_img_array)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in0, in1 = in0.to(device), in1.to(device)

    lpips_alex = lpips_alex.to(device)
    lpips_score = lpips_alex(in0, in1)
    return lpips_score.item()


def count_CLIP_I(test_img, origin_img, model, processor):
    # torch.cuda.empty_cache()

    data_pro = processor(images=[test_img, origin_img], return_tensors="pt", padding=True).to(model.device)
    data_pro = model.get_image_features(**data_pro)
    d = F.cosine_similarity(data_pro[0, :].unsqueeze(0), data_pro[1, :].unsqueeze(0))
    return d.item()

@torch.no_grad()
class Emo_S():
    def __init__(self):
        self.origin_image_pred = {}
        self.classifier = self.init_emo_cls()
        self.CLIPmodel = CLIPModel.from_pretrained("/mnt/d/model/clip-vit-base-patch32").to('cuda') #TODO
        self.processor = CLIPProcessor.from_pretrained("/mnt/d/model/clip-vit-base-patch32") #TODO
        self.emotion_list_8 = {"amusement": 0,
                               "awe": 1,
                               "contentment": 2,
                               "excitement": 3,
                               "anger": 4,
                               "disgust": 5,
                               "fear": 6,
                               "sadness": 7}

    def init_emo_cls(self, weight="Metric/weight/2024-02-23-best.pth"): #TODO
        classifier = clip_classifier(8).to('cuda')
        state = torch.load(weight, map_location='cuda')
        classifier.load_state_dict(state)
        classifier.eval()
        return classifier

    @torch.no_grad()
    def pred_emo(self, img):
        img = img.convert('RGB')
        data = self.processor(images=img, return_tensors="pt", padding=True).to('cuda')
        clip = self.CLIPmodel.get_image_features(**data)
        pred = self.classifier(clip.to('cuda'))
        return pred.cpu()

    @torch.no_grad()
    def count_emo_s(self, test_img, origin_img, origin_img_name, target_emo):
        if origin_img_name not in self.origin_image_pred:
            self.origin_image_pred[origin_img_name] = self.pred_emo(origin_img)
        test_emo = self.pred_emo(test_img)
        emo_index = self.emotion_list_8[target_emo]
        emo_s = test_emo.view(-1)[emo_index].item() - self.origin_image_pred[origin_img_name].view(-1)[emo_index].item()
        return emo_s


@torch.no_grad()
class DINO_extractor():
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("/mnt/d/model/dinov2-base") #TODO
        self.model = AutoModel.from_pretrained("/mnt/d/model/dinov2-base").to('cuda') #TODO
        self.origin_image_dino = {}

    @torch.no_grad()
    def extract_dino(self,img):
        inputs = self.processor(images=[img], return_tensors="pt").to('cuda')
        outputs = self.model(**inputs)
        feature = outputs.last_hidden_state.mean(dim=1)
        feature = feature.unsqueeze(1)
        return feature[0]

    @torch.no_grad()
    def count_dino_i(self,test_img, origin_img, origin_img_name):
        if origin_img_name not in self.origin_image_dino:
            self.origin_image_dino[origin_img_name] = self.extract_dino(origin_img).cpu()
        dino_sim = F.cosine_similarity(self.origin_image_dino[origin_img_name], self.extract_dino(test_img).cpu()).item()
        return dino_sim

def main(origin_img_dir, target_img_dir):
    classifer = Emo_S()
    dino_Ex = DINO_extractor()
    lpips_alex = lpips.LPIPS(net='alex')
    model = CLIPModel.from_pretrained("/mnt/d/model/clip-vit-base-patch32").to('cuda') #TODO
    processor = CLIPProcessor.from_pretrained("/mnt/d/model/clip-vit-base-patch32") #TODO
    test_img_paths = []
    for root, _, file_path in os.walk(target_img_dir):
        for file in file_path:
            if file.endswith("png") or file.endswith("jpg"):
                test_img_paths.append(os.path.join(root, file))
    img_metrics = {
        'SSIM': [],
        'PSNR': [],
        'LPIPS': [],
        'MSE': [],
        'CLIP-I': [],
        'DINO-I': [],
        'Emo-S': []
    }
    for test_img_path in tqdm(test_img_paths):
        test_img_name = test_img_path.split('/')[-1].split('_')[0]
        target_emotion = test_img_path.split('/')[-1].split('.')[0].split('_')[1]

        for suffix in ['jpg','png']:
            origin_img_path = os.path.join(origin_img_dir, f"{test_img_name}.{suffix}")

            if os.path.isfile(origin_img_path) is True:
                break
        with Image.open(test_img_path) as test_img, Image.open(origin_img_path) as origin_img:
            origin_img = crop_img(origin_img)
            test_img = test_img.resize((512, 512))
            test_img_array = np.array(test_img)
            origin_img_array = np.array(origin_img)

            ssim = count_ssim(test_img_array, origin_img_array)
            img_metrics['SSIM'].append(ssim)

            psnr = count_psnr(test_img_array, origin_img_array)
            img_metrics['PSNR'].append(psnr)

            current_mse = count_mse(test_img_array, origin_img_array)
            img_metrics['MSE'].append(current_mse)

            current_lpips = count_lpips(test_img_array, origin_img_array, lpips_alex)
            img_metrics['LPIPS'].append(current_lpips)

            current_clip_i = count_CLIP_I(test_img, origin_img, model, processor)
            img_metrics['CLIP-I'].append(current_clip_i)

            current_DINO = dino_Ex.count_dino_i(test_img_array, origin_img_array, test_img_name)
            img_metrics['DINO-I'].append(current_DINO)

            current_Emo_S = classifer.count_emo_s(test_img=test_img,
                                                  origin_img=origin_img,
                                                  origin_img_name=test_img_name,
                                                  target_emo=target_emotion)
            img_metrics['Emo-S'].append(current_Emo_S)

    averages = {key: sum(values) / len(values) if values else 0 for key, values in img_metrics.items()}
    with open(os.path.join(target_img_dir,'img_metrics_averages.txt'), 'a') as file:
        for key, avg in averages.items():
            file.write(f"{key}: {avg:.6f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, action='append', default=[1])
    parser.add_argument('--datadir', type=str, default=None)
    args = parser.parse_args()
    steps = args.step
    origin_img_dir = "/mnt/d/dataset/EmoEdit-inference-set/Original_405/" #TODO
    if args.datadir is None:
        target_img_dir = '/mnt/d/code/EmoEdit-official/train_data/test/validation-on-TrainSet-all-30' #TODO
    else:
        target_img_dir = args.datadir
    for step in steps:
        target_img_path = os.path.join(target_img_dir, str(step))
        main(origin_img_dir=origin_img_dir, target_img_dir=target_img_path)

