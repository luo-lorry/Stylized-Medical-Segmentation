import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from PIL import Image
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots-Skin/PraNet(style_transfer)/PraNet-39.pth')

for _data_name in ['test']:
    data_path = './Dataset/Dataset_Skin/{}/'.format(_data_name)
    save_path = './results/PraNet/{}-Skin/results(style_transfer)/'.format(_data_name)
    opt = parser.parse_args()
    model = PraNet()
    model.load_state_dict(torch.load(opt.pth_path, weights_only=True))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/style(original)/'.format(data_path, _data_name)
    gt_root = '{}/masks/'.format(data_path, _data_name)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    print(f"\nProcessing {_data_name}(style_transfer) images...")
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        im = Image.fromarray((res * 255).astype(np.uint8))
        im.save(save_path + name)

        if (i + 1) % 10 == 0:
            print(f'[{_data_name}(style_transfer)] Processed {i + 1}/{test_loader.size} images')

print("\nTesting complete!")
print("\nTesting complete!")