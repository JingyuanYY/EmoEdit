# EmoEdit: Evoking Emotions through Image Manipulation
> [Jingyuan Yang](https://jingyuanyy.github.io/), Jiawei Feng, Weibin Luo, Dani Lischinski2
, Daniel Cohen-Or, [Hui Huang*](https://vcc.tech/~huihuang)  
> Shenzhen University  
> Affective Image Manipulation (AIM) seeks to modify user-provided images to evoke specific emotional responses. This task is inherently complex due to its twofold objective: significantly evoking the intended emotion, while preserving the original image composition. Existing AIM methods primarily adjust color and style, often failing to elicit precise and profound emotional shifts. Drawing on psychological insights, we introduce EmoEdit, which extends AIM by incorporating content modifications to enhance emotional impact. Specifically, we first construct EmoEditSet, a large-scale AIM dataset comprising 40,120 paired data through emotion attribution and data construction. To make existing generative models emotion-aware, we design the Emotion adapter and train it using EmoEditSet. We further propose an instruction loss to capture the semantic variations in data pairs. Our method is evaluated both qualitatively and quantitatively, demonstrating superior performance compared to existing state-of-the-art techniques. Additionally, we showcase the portability of our Emotion adapter to other diffusion-based models, enhancing their emotion knowledge with diverse semantics.

<a href="https://arxiv.org/abs/2405.12661"><img src="https://img.shields.io/badge/arXiv-2405.12661-b31b1b.svg" height=22.5></a>

<p align="left">
<img src="docs/teaser.png" width="1200px"/>  
<br>
Fig 1. Affective Image Manipulation with EmoEdit, which seeks to modify a user-provided image to evoke specific emotional responses
in viewers. Our method requires only emotion words as prompts, without necessitating detailed descriptions of the input or output image.
</p>

## Description
Official implementation of our EmoEdit paper.

## Construction of EmoEditSet
<p align="left">
<img src="docs/method-1.png" width="1200px"/>  
<br>
Fig 2. Overview of EmoEditSet. (a) Emotion factor trees are built with various representative semantic summaries based on EmoSet.
(b) Through careful collection, generation and filtering, EmoEditSet is built with high-quality and semantic-diverse paired data.
</p>

## Setup
To create the conda environment needed to run the code, run the following command:

```
conda env create -f environment/env.yaml
conda activate EmoGen
```

Alternatively, install the requirements from `requirements.txt`

## Usage

### Preliminary
[EmoSet](https://vcc.tech/EmoSet) is needed to train in this network, as attribute label is necessary.

We need to organize the dataset according to its attributes, and the following is its layout:

```
data_root
|
├── object
|    ├── (3) cart
|    |    ├── disgust_05770.jpg
|    |    ├── ...
|    |    └── sadness_10803.jpg
|    ├── ...
|    └── (13094) plant
|
└── scene
     ├── (1) airfield
     ├── ...
     └── (2406) street
```
The number before the attribute represents the total number of images with this attribute.
### Training
To train our network, follow these steps:

First, manually modify the code related to reading EmoSet and change the file location to the location where your EmoSet is located. For example:
In training/dataset_balance.py
```
annotion_path = f'/mnt/d/dataset/EmoSet/annotation/{emotion}/{emotion}_{number}.json' # change to "{your_EmoSet_location}/annotation/{emotion}/{emotion}_{number}.json"
```

Secondly, create training dataset:
```
python training/dataset_balance.py
```

Thirdly, start to train your own network:
```
accelerate training/main.py
```

Finally, generate emotional image:
```
python training/inference.py
```
You can modify config/config.yaml to change some details.

### Emotion Creation

<p align="left">
<img src="docs/exp-5.png" width="1500px"/>  
<br>
Fig 3. Emotion creation. (a) transfers emotion representations (i.e., amusement, fear) to a series of neutral contents while (b) fuse two emotions (i.e., amusement-awe, amusement-fear) together, which may be helpful for emotional art design.
</p>

#### Emotion Transfer
To transfer emotion into object, follow these steps:
First, change training/inference.py code:
```
use_prompt = True
generate(output_dir, device, model, num_fc_layers, need_LN, need_ReLU, need_Dropout, use_prompt)
```

Then, you can choose your object:
```
templates = [
      "{} bag", "{} cup", "{} room", "{} street",
]
```

```
python training/inference.py
```

#### Emotion Fusion
to fuse different emotion together, follow these steps:

```
python training/inference_combine_emotion.py
```
this code has similar structure as training/inference.py.

## Results
#### Comparison with other diffusion models
<p align="left">
<img src="docs/exp-1.png" width="1000px"/>  
<br>
Fig 4. Qualitative comparisions with the state-of-the-art text-to-image generation approaches and ablation studies of our method.
</p>

<div align="center">
     
Table 1. Comparisons with the state-of-the-art methods on emotion generation task, involving five metrics.
| Method | FID &darr; | LPIPS &uarr; | Emo-A &uarr; | Sem-C &uarr; | Sem-D &uarr; |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Stable Diffusion | 44.05 | 0.687 | 70.77% | 0.608 | 0.0199 |
| Textual Inversion | 50.51 | 0.702 | 74.87% | 0.605 | 0.0282 |
| DreamBooth | 46.89| 0.661 | 70.50% | 0.614 | 0.0178 |
| Ours     | **41.60** | **0.717** | **76.25%** | **0.633** | **0.0335** |

</div>

<div align="center">

Table 2.  User preference study. The numbers indicate the percentage of participants who prefer our results over those compared
methods, given the same emotion category as input.
| Method | Image fidelity &uarr; | Emotion faithfulness &uarr; | Semantic diversity &uarr; |
|:-------:|:-------:|:-------:|:-------:|
| Stable Diffusion | 67.86±15.08% | 73.66±11.80% | 87.88±9.64% |
| Textual Inversion | 79.91±16.92% | 72.75±16.90% | 85.66±10.51% |
| DreamBooth | 77.23±14.00% | 80.79±8.64% | 81.68±17.06% |

</div>

## Citation
If you find this work useful, please kindly cite our paper:
```
@article{yang2024emogen,
  title={EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models},
  author={Yang, Jingyuan and Feng, Jiawei and Huang, Hui},
  journal={arXiv preprint arXiv:2401.04608},
  year={2024}
}
```