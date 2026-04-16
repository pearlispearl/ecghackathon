import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class ECGImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.copy()
        self.image_dir = image_dir
        self.transform = transform

        def cac_to_class(score):
            if score == 0:
                return 0
            elif 1 <= score <= 99:
                return 1
            elif 100 <= score <= 399:
                return 2
            elif 400 <= score <= 1000:
                return 3
            else: # > 1000
                return 4

        self.df["label"] = self.df["CaScore"].apply(cac_to_class)

        # mapping file name between .csv and .png
        self.image_map = {fname.split("_")[0]: fname for fname in os.listdir(image_dir) if "_" in fname}

        # filter ou NaN value
        self.df["HN_str"] = self.df["HN"].astype(str).str.zfill(7)
        self.df = self.df[self.df["HN_str"].isin(self.image_map.keys())].reset_index(drop=True)
        
        print(f"Dataset initialized: {len(self.df)} samples found.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        hn_str = row["HN_str"]

        img_path = os.path.join(self.image_dir, self.image_map[hn_str])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.long)

        return image, label
