import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.UNetPP import NestedUNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch, opt, total_step):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record_output, loss_record_attention = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            output, attention = model(images)
            # ---- loss function ----
            loss_output = structure_loss(output, gts)
            loss_attention = structure_loss(attention, gts)
            loss = 1 * loss_output + 0.5 * loss_attention
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record_output.update(loss_output.data, opt.batchsize)
                loss_record_attention.update(loss_attention.data, opt.batchsize)
        # ---- train(cycle) visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[Output Loss: {:.4f}, Attention Loss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record_output.show(), loss_record_attention.show()))
    # ---- save model ----
    save_path = 'snapshots-Polyp/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'UNetPP-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'UNetPP-%d.pth' % epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=30, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=8e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=12, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=256, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_save', type=str,
                        default='UNetPP(style_transfer)', help='path to save trained models')
    opt = parser.parse_args()

    # ---- build models ----
    model = NestedUNet(in_channels=3, out_channels=1, base_channels=32, pretrained=True).cuda()

    # ---- optimizer ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    # ---- data loader ----
    image_root = './Dataset/Dataset_Polyp/train/style(ST)/'
    gt_root = './Dataset/Dataset_Polyp/train/masks/'
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt, total_step)

