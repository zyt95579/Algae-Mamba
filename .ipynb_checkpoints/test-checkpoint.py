import random
from torch.utils.data import DataLoader
from dataloaders.dataset import BaseDataSets
import argparse
import os
import shutil
from val_2D import compute_mIoU
import numpy as np
import torch
from PIL import Image
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                     default='../data/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='gf', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--val_path', type=str,
                    default='../img/', help='Name of Experiment')
parser.add_argument('--mode_path', type=str,
                    default='/home/wsp/桌面/Mamba-UNet-main/model/gf_140_labeled/mambaunet/iter_21000_mIoU_72.78.pth', help='Name of Experiment')

def Inference(FLAGS):
    db_val = BaseDataSets(base_dir=FLAGS.root_path, exp=FLAGS.exp, split="test")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    net = net_factory(net_type='mambaunet', class_num=FLAGS.num_classes)
    net.load_state_dict(torch.load(FLAGS.mode_path))
    print("init weight from {}".format(FLAGS.mode_path))
    net.eval()
    for i_batch, sampled_batch in enumerate(valloader):
        out = net(sampled_batch["image"].cuda())
        out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy().astype(np.uint8)
        idx = sampled_batch["idx"]
        name = valloader.dataset.label_list[idx]
        name = name.split('/')[-1]
        pred_dir = os.path.join(FLAGS.val_path, name)
        image = Image.fromarray(out)  # 将NumPy数组转换为PIL图像
        image.save(pred_dir)
    gt_dir = os.path.join(FLAGS.root_path, FLAGS.exp, 'test/label')
    image_ids = open('/home/wsp/桌面/Mamba-UNet-main/data/val.txt').read().splitlines()
    name_classes = ["_background_", "HT", "MWZ"]
    hist, IoUs, Recall, Precision, mIoU = compute_mIoU(gt_dir, FLAGS.val_path, image_ids, FLAGS.num_classes, name_classes)

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    hist, IoUs, Recall, Precision, mIoU = Inference(FLAGS)

