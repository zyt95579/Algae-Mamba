import argparse
import logging
import os
import random
import shutil
import sys
import time
from val_2D import compute_mIoU
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from networks.vision_mamba import MambaUnet as VIM_seg
from config import get_config
from dataloaders.dataset import BaseDataSets
from utils import losses


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='gf', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--val_path', type=str,
                    default='../img/', help='Name of Experiment')
parser.add_argument('--in_chans', type=int, default=6,
                    help='input channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')


parser.add_argument('--max_iterations', type=int,
                    default=40000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=140,
                    help='labeled data')
args = parser.parse_args()


config = get_config(args)

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    model = VIM_seg(config, img_size=args.patch_size, num_classes=args.num_classes).cuda()
    model.load_from(config)
    db_train = BaseDataSets(base_dir=args.root_path, exp=args.exp, split="train", )
    db_val = BaseDataSets(base_dir=args.root_path, exp=args.exp, split="test")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # image = volume_batch[1]
            # band_r, band_g, band_b, band_nir = image[0, :, :], image[1, :, :], image[2, :, :], image[3, :, :]
            # eps = 1e-10
            # ndvi = (band_nir - band_r) / (band_r + band_nir + eps)
            # ndvi = ndvi.cpu().numpy()
            # ndvi_image = Image.fromarray((ndvi * 255).astype('uint8'))
            # ndvi_image.show()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            label_batch = label_batch * 255
            label_batch = label_batch.squeeze(1)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            # if iter_num % 20 == 0:
            #     image = volume_batch[1, 0:1, :, :]
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(
            #         outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction',
            #                      outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 3000 == 0:
                model.eval()
                for i_batch, sampled_batch in enumerate(valloader):

                    # image = sampled_batch["image"]
                    # image = volume_batch[1]
                    # band_r, band_g, band_b, band_nir = image[0, :, :], image[1, :, :], image[2, :, :], image[3, :, :]
                    # eps = 1e-10
                    # ndvi = (band_nir - band_r) / (band_r + band_nir + eps)
                    # ndvi = ndvi.cpu().numpy()
                    # ndvi_image = Image.fromarray((ndvi * 255).astype('uint8'))
                    # ndvi_image.show()

                    out = model(sampled_batch["image"].cuda())
                    out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
                    out = out.cpu().detach().numpy().astype(np.uint8)
                    idx = sampled_batch["idx"]
                    name = valloader.dataset.label_list[idx]
                    name = name.split('/')[-1]
                    pred_dir = os.path.join(args.val_path, name)
                    image = Image.fromarray(out)  # 将NumPy数组转换为PIL图像
                    image.save(pred_dir)
                gt_dir = os.path.join(args.root_path, args.exp, 'test/label')
                image_ids = open('/home/wsp/桌面/Mamba-UNet-main/data/val.txt').read().splitlines()
                name_classes = ["_background_", "HT", "MWZ"]
                hist, IoUs, Recall, Precision, mIoU = compute_mIoU(gt_dir, args.val_path, image_ids, num_classes, name_classes)
                if mIoU > best_performance:
                    best_performance = mIoU
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_mIoU_{}.pth'.format(iter_num, round(best_performance, 4)))
                    torch.save(model.state_dict(), save_mode_path)
                model.train()
            #
            # if iter_num % 3 == 0:
            #     save_mode_path = os.path.join(
            #         snapshot_path, 'iter_' + str(iter_num) + '.pth')
            #     torch.save(model.state_dict(), save_mode_path)
            #     logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
