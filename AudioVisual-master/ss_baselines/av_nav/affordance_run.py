import os
import torch
import pickle
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from PIL import Image
from torch.nn import functional as F
from ipdb import set_trace
import argparse
from tqdm import tqdm

from affordance.networks import AffordanceModuleShallow, AffordanceModuleBottleNeck
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True


def json2point(input_shape, c_x, c_y, sigma):
    img_height = input_shape[0]
    img_width = input_shape[1]
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)

    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


class CustomDataset(Dataset):
    def __init__(self, root_file_path, trans, trans_heat):
        with open(root_file_path, 'r') as f:
            self.data_paths = [line[:-1] for line in f]
        # overfit 一个房间的多个sample
        # with open(root_file_path, 'r') as f:
        #     data_paths = [line[:-1] for line in f]
        # self.data_paths = []
        # for path in data_paths:
        #     if 'office_4' in path:
        #         self.data_paths.append(path)
        self.transform = trans
        self.trans_heat = trans_heat

    def __getitem__(self, index):
        path = self.data_paths[index]
        path_meta = path + '_meta.pickle'
        path_rgb = path + '_rgb.png'
        path_heatmap_np = path + '_heatmap.np'
        with open(path_meta, 'rb') as f:
            metas = pickle.load(f)
        with open(path_heatmap_np, 'rb') as f:
            heatmap = pickle.load(f)
        orientation = int(metas['orientation'] // 90)
        rgb = Image.open(path_rgb).convert('RGB')
        processed_rgb = self.transform(rgb)
        # [0, 255] -> [-1, 1]
        processed_rgb = processed_rgb * 2 - 1
        # *100 是防止小东西被插值搞没了，反正后面会normailize回去的
        heatmap = self.trans_heat(Image.fromarray(heatmap*100))
        # 得norm成一个分布，不然没法CE loss
        heatmap /= heatmap.sum()
        audio_spec = metas['audio']
        # set_trace()
        return processed_rgb, heatmap, torch.Tensor(audio_spec), torch.LongTensor([orientation])

    def __len__(self):
        return len(self.data_paths)


def render_color_heat(heat, image):
    heat = heat.permute(1, 2, 0).detach().cpu().numpy()
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    image = (image+1)*255 / 2
    # 后面的热力图也会转化为RGB 所以现在不用cvtcolor
    image = image.astype(np.uint8)
    # 先norm这张heatmap
    gray_img = heat
    norm_img = np.zeros(gray_img.shape)
    norm_img = cv2.normalize(gray_img, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    # 搞个热力图
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
    img_add = cv2.addWeighted(image, 0.3, heat_img, 0.7, 0)

    # 处理回去
    img_add = img_add.astype(np.float32)
    # [0, 255] -> [0, 1]
    img_add = torch.tensor(img_add/255.).permute(2, 0, 1)
    return img_add.unsqueeze(0)


def visualize_old(writer, test_image, preds, test_heat):
    contrast = []
    for pred, gt in zip(preds, test_heat):
        pred = pred.unsqueeze(0).unsqueeze(0)
        pred /= pred.max()
        contrast.append(pred)
        gt = gt.unsqueeze(0)
        gt /= gt.max()
        contrast.append(gt)
    contrast = torch.cat(contrast, dim=0)
    pred_grid = make_grid(preds.unsqueeze(1), nrow=8)
    pred_grid /= pred_grid.max()
    gt_grid = make_grid(test_heat, nrow=8)
    gt_grid /= gt_grid.max()
    mix = pred_grid + gt_grid
    mix /= mix.max()
    # contrast /= contrast.max()
    writer.add_image('eval/preds', pred_grid, epoch)
    writer.add_image('eval/images', make_grid((test_image + 1) * 255 / 2, nrow=8), epoch)
    writer.add_image('eval/heats', gt_grid, epoch)
    writer.add_image('eval/contrast', make_grid(contrast, nrow=8), epoch)
    writer.add_image('eval/mix', mix, epoch)


def visualize_new(writer, test_image, preds, test_heat, test_audio):
    with torch.no_grad():
        # 先把gt搞到test image上
        gt_color = []
        for gt, raw in zip(test_heat, test_image):
            gt_color.append(render_color_heat(gt, raw))
        gt_color = torch.cat(gt_color, dim=0)
        # 再把pred搞到test image上
        pred_color = []
        for pred, raw in zip(preds.unsqueeze(1), test_image):
            pred_color.append(render_color_heat(pred, raw))
        pred_color = torch.cat(pred_color, dim=0)
        # 再搞一张对比图
        contrast = []
        for gt, pred in zip(gt_color, pred_color):
            contrast.append(gt.unsqueeze(0))
            contrast.append(pred.unsqueeze(0))
        contrast = torch.cat(contrast, dim=0)
        # 声音也可视化一下
        test_audio_0 = test_audio[:, :, :, 0:1].permute(0, 3, 1, 2)
        test_audio_1 = test_audio[:, :, :, 1:2].permute(0, 3, 1, 2)
        max_0, _ = torch.max(test_audio_0.view(preds.shape[0], -1), dim=1)
        max_1, _ = torch.max(test_audio_1.view(preds.shape[0], -1), dim=1)
        test_audio_0 /= max_0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        test_audio_1 /= max_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        writer.add_image('eval/gt_color_heats', make_grid(gt_color, nrow=8), epoch)
        writer.add_image('eval/pred_color_heats', make_grid(pred_color, nrow=8), epoch)
        writer.add_image('eval/contrast', make_grid(contrast, nrow=8), epoch)
        writer.add_image('eval/test_audio_1', make_grid(test_audio_1, nrow=8), epoch)
        writer.add_image('eval/test_audio_0', make_grid(test_audio_0, nrow=8), epoch)
        writer.add_image('eval/raw_img', make_grid((test_image+1)/2, nrow=8), epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--data_exp', type=str, default='debug')
    args = parser.parse_args()
    exists_or_mkdir(f'data/models/replica/{args.exp_name}')
    exists_or_mkdir(f'data/models/replica/{args.exp_name}/tb')
    log_path = f'data/models/replica/{args.exp_name}/tb'
    # init writer
    writer = SummaryWriter(log_path)
    # init dataset
    data_path = f'./Dataset/{args.data_exp}/paths.txt'
    img_size = 128
    heat_size = 128
    batch_size = 64
    epoch_num = 200
    ch = 64
    audio_dim = 128
    feat_dim = 128
    ori_dim = 64
    lr = 4e-4
    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor()])
    transform_heat = transforms.Compose([transforms.Resize((heat_size, heat_size)),
                                         transforms.ToTensor()])
    train_dataset = CustomDataset(data_path, trans=transform, trans_heat=transform_heat)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # init model
    model = AffordanceModuleBottleNeck(ch=ch, audio_dim=audio_dim, ori_dim=ori_dim, feat_dim=feat_dim)
    model.to(device)
    model.train()

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # init loss func
    loss_func = nn.CrossEntropyLoss()

    # init test batch
    test_batch = next(iter(train_loader))
    test_image, test_heat, test_audio, test_ori = test_batch
    test_image = test_image.to(device)
    test_audio = test_audio.to(device)
    test_heat = test_heat.to(device)
    test_ori = test_ori.to(device)

    for epoch in tqdm(range(epoch_num)):
        if epoch % 1 == 0:
            with torch.no_grad():
                # eval model
                model.eval()
                preds = model(test_image, test_audio, test_ori)
                visualize_new(writer, test_image, preds, test_heat, test_audio)
                model.train()
        # for step, batch in enumerate(train_loader):
        for step, _ in enumerate(range(100)):
            # 测试一个batch的overfit
            batch = test_batch
            image, heat, audio, ori = batch
            image = image.to(device)
            audio = audio.to(device)
            heat = heat.to(device)  # gt
            ori = ori.to(device)
            bs = image.shape[0]
            pred = model(image, audio, ori)
            # set_trace()
            # loss = torch.nn.functional.kl_div(torch.log(pred.view(bs, -1)), heat.view(bs, -1), reduction='batchmean')
            # loss = loss_func(pred.view(bs, -1), heat.view(bs, -1))
            # loss = torch.mean(-torch.log(heat)*torch.abs(pred-heat))
            # loss = torch.mean(torch.abs(pred-heat)**2)
            loss = torch.sum(torch.abs(pred-heat)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_loss', loss, epoch*len(train_loader)+step)
        scheduler.step()
