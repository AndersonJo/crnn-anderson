import os

from PIL import Image
from skimage import io

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np


class LicensePlateDataset(Dataset):
    def __init__(self, dir_path: str, transform=None):

        self.dir_path = dir_path
        self.transform = transform
        self.img_files = [f for f in os.listdir(self.dir_path) if f.endswith('jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        img_path = os.path.join(self.dir_path, file_name)
        image = Image.fromarray(io.imread(img_path))

        if self.transform is not None:
            image = self.transform(image)

        y = file_name[:-4]

        return image, y

    def calculate_mean_and_std(self):
        images = [self[i][0] for i in tqdm(range(len(self)), desc='load')]
        mean = np.concatenate([[i.mean((0, 1))] for i in tqdm(images, desc='mean')]).mean(0)
        std = np.concatenate([[i.std((0, 1))] for i in tqdm(images, desc='std')]).mean(0)
        print('mean:', mean)
        print('std:', std)

    def move_validatation_files(self, test_size=0.3):
        if not os.path.exists('valid_data'):
            os.mkdir('valid_data')

        n = len(self)
        n_sample = int(n * 0.3)
        sample_files = np.random.choice(self.img_files, size=n_sample, replace=False)
        for file_name in sample_files:
            from_path = os.path.join(self.dir_path, file_name)
            to_path = os.path.join('valid_data', file_name)
            os.rename(from_path, to_path)
