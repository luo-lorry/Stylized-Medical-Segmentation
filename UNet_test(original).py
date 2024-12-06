import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from PIL import Image
from lib.UNet import UNet
from utils.dataloader import test_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=256, help='Testing size')
    parser.add_argument('--pth_path', type=str, default='./snapshots-Skin/UNet(original)/UNet-29.pth',
                        help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, default='./results/UNet/', help='Directory to save results')
    args = parser.parse_args()

    data_names = ['test']

    for _data_name in data_names:
        data_path = './Dataset/Dataset_Skin/{}/'.format(_data_name)
        save_path = './results/UNet/{}-Skin/results(original)/'.format(_data_name)
        image_root = '{}/style(original)/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        os.makedirs(save_path, exist_ok=True)

        model = UNet(in_channels=3, out_channels=1, base_channels=32, pretrained=False).cuda()
        model.load_state_dict(torch.load(args.pth_path))
        model.eval()

        test_loader = test_dataset(image_root, gt_root, args.testsize)

        print(f"\nProcessing {_data_name} images with UNet...")
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            with torch.no_grad():
                output, attention = model(image)
                res = output
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            im = Image.fromarray((res * 255).astype(np.uint8))
            im.save(os.path.join(save_path, name))

            if (i + 1) % 10 == 0:
                print(f'[{_data_name}] Processed {i + 1}/{test_loader.size} images')

    print("\nTesting complete!")


if __name__ == '__main__':
    main()