from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms

class ImageDataset(torch.utils.data.Dataset):
    
    def __init__(self, mode, image_size, file_loc="data/train_new.csv", 
                    shuffle=True, fold=0, do_transform=True):
        self.mode = mode
        self.image_size = image_size
        self.do_transform = do_transform

        df = pd.read_csv(file_loc)
        cols_dict = dict(df.dtypes)
        cols_dict = {k:str(v) for k, v in cols_dict.items()}
        self.target_cols = [x for x in cols_dict.keys() if cols_dict[x] == 'int64' and x != 'fold']
        self.img_folder = "train" if "fold" in df.columns else "test"

        if self.mode == "train":
            self.df = df[df["fold"] != fold]
        else:
            if self.img_folder == "train":
                self.df = df[df["fold"] == fold]
            else:
                self.df = df

        if shuffle:
            self.df = self.df.sample(frac=1.0)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomGrayscale(p=1),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomAffine(10, translate=None, scale=None, 
                                    shear=None, resample=0, fillcolor=0),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.1),
            transforms.RandomRotation(10, resample=False, expand=False, 
                                    center=None, fill=None),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                                    saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), 
                                    ratio=(0.3, 3.3), inplace=True),
        ])
        self.weak_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomGrayscale(p=1),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_path = Path(f"data/{self.img_folder}/{row['StudyInstanceUID']}.jpg")
        img = Image.open(img_path)
        if self.mode == "train" and self.do_transform:
            img = self.img_transform(img)
        else:
            img = self.weak_transform(img)
        img = img.repeat(3,1,1)

        label = np.array(row[self.target_cols]).astype(np.float32)
        label = torch.from_numpy(label)

        img_name = row['StudyInstanceUID']

        return {
            "image"     : img, 
            "label"     : label,
            "img_name"  : img_name
        }

    def __len__(self):
        return len(self.df)

if __name__ == "__main__":
    sample_dataset = ImageDataset("test", 256)

    fig, ax = plt.subplots(2, 4)
    for i in range(8):
        imgdict = sample_dataset[i]
        ax[i // 4][i % 4].imshow(
            imgdict["image"].permute(1,2,0).numpy()[:,:,0]
        )

    plt.show()