import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class ECGWaveToImageDataset(Dataset):
    def __init__(self, df, wave_dict, transform=None):
        self.df = df.copy()
        self.wave_dict = wave_dict
        self.transform = transform

        def cac_to_class(score):
            if score == 0: return 0
            elif score <= 99: return 1
            elif score <= 399: return 2
            elif score <= 1000: return 3
            else: return 4

        self.df["label"] = self.df["CaScore"].apply(cac_to_class)
        self.df["id_str"] = self.df["match_id"].astype(str)

        # keep only rows with waveform
        self.df = self.df[self.df["id_str"].isin(self.wave_dict.keys())].reset_index(drop=True)

        print(f"Wave dataset: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wave = self.wave_dict[row["id_str"]]  # shape (12, T)

        # ---- convert to image ----
        fig = plt.figure(figsize=(4, 4))
        for i in range(min(12, len(wave))):
            plt.plot(wave[i])
        plt.axis("off")

        fig.canvas.draw()

        img = np.asarray(fig.canvas.renderer.buffer_rgba())
        # remove alpha channel (RGBA → RGB)
        img = img[:, :, :3]

        plt.close(fig)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row["label"], dtype=torch.long)

        return img, label