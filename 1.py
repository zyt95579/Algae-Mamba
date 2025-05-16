import random
from torch.utils.data import DataLoader

import os
from PIL import Image
import numpy as np
from dataloaders.dataset import BaseDataSets
import argparse
import os
import shutil
from val_2D import compute_mIoU
import numpy as np
import torch
from PIL import Image
from networks.net_factory import net_factory


folder1 = '/home/wsp/下载/img2'
folder2 = '/home/wsp/下载/img2'
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)
common_files = set(files1) & set(files2)
image_pairs = {}
for file in common_files:
    path1 = os.path.join(folder1, file)
    path2 = os.path.join(folder2, file)
    image1 = Image.open(path1)
    image2 = Image.open(path2)
    np_image1 = np.array(image1)
    np_image2 = np.array(image2)
    image_pairs[file] = (np_image1, np_image2)

# 现在 image_pairs 字典包含了相同文件名的图像数组对
gt_dir = '/home/wsp/桌面/Mamba-UNet-main/data/gf/test/label'
val_path = '/home/wsp/桌面/Mamba-UNet-main/data/val.txt'
num_classes = 3
image_ids = open('/home/wsp/桌面/Mamba-UNet-main/data/val.txt').read().splitlines()
hist, IoUs, Recall, Precision, mIoU = compute_mIoU(gt_dir, val_path, image_ids, num_classes)


