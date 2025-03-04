import pandas as pd
import itertools
from torch.utils.data import Dataset
import os
import utils



class GenerateDataset(Dataset):
    def __init__(
            self,
            image_dir,
            summary_path,
    ):
        self.image_dir = image_dir

        image_paths = utils.get_all_paths(self.image_dir, ['jpg', 'png'])

        df = pd.read_csv(summary_path, header=0)
        summary_list = df['3 words (v1)'].tolist()
        emotion_list = df['emotion'].tolist()
        new_list = zip(summary_list, emotion_list)
        self.pairs = list(itertools.product(image_paths, new_list))

        self.num_images = len(image_paths)
        self._length = len(self.pairs)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        path, (summary, emotion) = self.pairs[i % self._length]
        example['path'] = path
        example['summary'] = summary
        example['emotion'] = emotion
        return example