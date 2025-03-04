from datetime import datetime
from torchvision import transforms
import torch
import os
import argparse
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
from tqdm import tqdm
from dataset import GenerateDataset
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

# 获取当天日期
today = datetime.now().strftime("%Y%m%d")


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
    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize(output_size)


    return img_resized

def load_images_from_paths(image_paths):
    images = []
    for path in image_paths:
        if os.path.exists(path): 
            try:
                img = Image.open(path).convert("RGB") 
                img = crop_image(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        else:
            print(f"Image path {path} does not exist.")
    
    return images

# Emotion = ["amusement","awe","contentment","excitement","anger","disgust","fear","sadness"]
parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='/mnt/d/data/EmoEdit/unsplash_05-07') #TODO
parser.add_argument('--source_dir', type=str, default='/mnt/d/dataset/unsplash_05-07') #TODO
parser.add_argument('--seed', type=int, help='seed', default=250)
parser.add_argument('--train_batch_size', type=int,  default=1)
parser.add_argument('--dataloader_num_workers', type=int,  default=8)
parser.add_argument('--image_guidance_scale', type=float, help='Image Preservation', default=1.5)
parser.add_argument('--guidance_scale', type=float, default=7.5)
args = parser.parse_args()

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.output_dir)
# writer = SummaryWriter(log_dir=args.output_dir)
accelerator = Accelerator(
    project_config=accelerator_project_config,
)

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


train_dataset = GenerateDataset(
    image_dir= args.source_dir,
    summary_path = f'/mnt/d/code/EmoEdit/CVPR/Summary_v8.csv',
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.dataloader_num_workers,
)


pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "/mnt/d/model/instruct-pix2pix", torch_dtype=weight_dtype, requires_safety_checker=False, safety_checker=None
).to(accelerator.device)
g_cuda = torch.Generator(device=accelerator.device)
if args.seed is not None:
    g_cuda.manual_seed(args.seed)
pipe.set_progress_bar_config(disable=True)
pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
pipe, train_dataloader = accelerator.prepare(
    pipe, train_dataloader
)

# emotion classifier
# weight = "emotion_cls/weight/2024-02-23-best.pth"
# classifier = Emotion_cls(weight, device)
progress_bar = tqdm(range(0, len(train_dataloader)), disable=not accelerator.is_local_main_process)
progress_bar.set_description("Steps")
for _, batch in enumerate(train_dataloader):

    image_paths = batch['path']
    summary = batch['summary']
    emotion = batch['emotion']
    if os.path.isfile(os.path.join(args.output_dir, f"{emotion[0]}/{image_paths[0].split('/')[-1].split('.')[0]}_{summary[0]}.jpg")):
        progress_bar.update(1)
        continue
    image = load_images_from_paths(image_paths)
    new_summary_list = ["add" + s for s in summary]
    generated_img = pipe(prompt=new_summary_list, image=image, guidance_scale=args.guidance_scale,
                               image_guidance_scale=args.image_guidance_scale,
                               generator=g_cuda).images
    for i, img in enumerate(generated_img):
        img_name = image_paths[i].split('/')[-1].split('.')[0]
        save_dir = os.path.join(args.output_dir, f"{emotion[i]}")
        os.makedirs(save_dir, exist_ok=True)
        img.save(os.path.join(args.output_dir, f"{emotion[i]}/{img_name}_{summary[i]}.jpg"))
    progress_bar.update(1)
