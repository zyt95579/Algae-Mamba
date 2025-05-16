import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch


class BaseDataSets(Dataset):
    def __init__(
            self,
            exp=None,
            base_dir=None,
            split="train",
    ):
        self.exp = exp
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        super(BaseDataSets, self).__init__()
        self.flag = "train" if self.split == "train" else "test"
        data_root = os.path.join(base_dir, self.exp, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        img_names = [i for i in os.listdir(os.path.join(data_root, "image")) if i.endswith(".tif")]
        label_names = [i.replace('.tif', '.png') for i in img_names]
        self.image_list = [os.path.join(data_root, "image", i) for i in img_names]
        self.label_list = [os.path.join(data_root, "label", i) for i in label_names]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx], cv2.IMREAD_UNCHANGED)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        label = Image.open(self.label_list[idx])
        image = transforms.ToTensor()(image)
        # eps = 1e-10
        # band_b, band_g, band_r, band_nir = image[0, :, :], image[1, :, :], image[2, :, :], image[3, :, :]
        # ndwi = (band_g - band_nir) / (band_g + band_nir + eps)
        # nli = ((band_nir * band_nir) - band_r) / ((band_nir * band_nir) + band_r + eps)
        # ndvi = (band_nir - band_r) / (band_nir + band_r + eps)
        # rvi = band_nir / (band_r + eps)
        # osavi =  (band_nir - band_r) / (band_nir + band_r + 0.16 + eps)
        # image = torch.cat((image, osavi.unsqueeze(0)), dim=0)
        label = transforms.ToTensor()(label)
        sample = {"image": image, "label": label}
        sample["idx"] = image.save(pred_dir)idx

        return sample
