from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms

class ImageDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_size, file_loc="data/train.csv", 
                    shuffle=True, do_transform=True):
        self.image_size = image_size
        self.do_transform = do_transform

        self.df_train = pd.read_csv(file_loc)
        cols_dict = dict(self.df_train.dtypes)
        cols_dict = {k:str(v) for k, v in cols_dict.items()}
        self.target_cols = [x for x in cols_dict.keys() if cols_dict[x] == 'int64']

        if shuffle:
            self.df_train = self.df_train.sample(frac=1.0)

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

    def __getitem__(self, index):
        row = self.df_train.iloc[index]
        img_path = Path(f"data/train/{row['StudyInstanceUID']}.jpg")
        img = Image.open(img_path)
        if self.do_transform:
            img = self.img_transform(img)

        label = np.array(row[self.target_cols]).astype(np.float32)
        label = torch.from_numpy(label)
        return img, label

    def __len__(self):
        return len(self.df_train)

if __name__ == "__main__":
    sample_dataset = ImageDataset(256)

    fig, ax = plt.subplots(2, 4)
    for i in range(8):
        (sample, label) = sample_dataset[i]
        ax[i // 4][i % 4].imshow(sample.permute(1,2,0).numpy()[:,:,0])

    plt.show()