import argparse
import datetime
import os
import random
import sys
import time
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchvision
import tqdm
from PIL import ImageFile

from utils import utils_aug
from utils.utils import (
    MetricDataset, Model_Inference, classification_metric, dict_to_PrettyTable,
    model_fuse, select_device, visual_predictions, visual_tsne
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.deterministic = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=r'dataset/train', help='train data path')
    parser.add_argument('--val_path', type=str, default=r'dataset/val', help='val data path')
    parser.add_argument('--test_path', type=str, default=r'dataset/test', help='test data path')
    parser.add_argument('--label_path', type=str, default=r'dataset/label.txt', help='label path')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--task', type=str, choices=['train', 'val', 'test', 'fps'], default='test', help='train, val, test, fps')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save path for model and log')
    parser.add_argument('--test_tta', action="store_true", help='using TTA Tricks')
    parser.add_argument('--visual', action="store_true", help='visual dataset identification')
    parser.add_argument('--tsne', action="store_true", help='visual tsne')
    parser.add_argument('--half', action="store_true", help='use FP16 half-precision inference')
    parser.add_argument('--model_type', type=str, choices=['torch', 'torchscript', 'onnx', 'tensorrt'], default='torch', help='model type(default: torch)')

    opt = parser.parse_known_args()[0]

    DEVICE = select_device(opt.device, opt.batch_size)
    if opt.half and DEVICE.type == 'cpu':
        raise ValueError('half inference only supported GPU.')
    if not os.path.exists(os.path.join(opt.save_path, 'best.pt')):
        raise FileNotFoundError('best.pt not found. please check your --save_path folder')
    ckpt = torch.load(os.path.join(opt.save_path, 'best.pt'))
    train_opt = ckpt['opt']
    set_seed(train_opt.random_seed)
    model = Model_Inference(DEVICE, opt)

    print(f"found checkpoint from {opt.save_path}, model type:{ckpt['model'].name}\n{dict_to_PrettyTable(ckpt['best_metric'], 'Best Metric')}")

    test_transform = utils_aug.get_dataprocessing_teststage(train_opt, opt, torch.load(os.path.join(opt.save_path, 'preprocess.transforms')))

    if opt.task == 'fps':
        inputs = torch.rand((opt.batch_size, train_opt.image_channel, train_opt.image_size, train_opt.image_size)).to(DEVICE)
        if opt.half and torch.cuda.is_available():
            inputs = inputs.half()
        warm_up, test_time = 100, 300
        fps_arr = []
        for i in tqdm.tqdm(range(test_time + warm_up)):
            since = time.time()
            with torch.inference_mode():
                model(inputs)
            if i > warm_up:
                fps_arr.append(time.time() - since)
        fps = np.mean(fps_arr)
        print(f'{fps:.6f} seconds, {1 / fps:.2f} fps, @batch_size {opt.batch_size}')
        sys.exit(0)
    else:
        save_path = os.path.join(opt.save_path, opt.task, datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S'))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        CLASS_NUM = len(os.listdir(getattr(opt, f'{opt.task}_path')))
        test_dataset = MetricDataset(torchvision.datasets.ImageFolder(getattr(opt, f'{opt.task}_path'), transform=test_transform))
        test_dataset = torch.utils.data.DataLoader(test_dataset, opt.batch_size, shuffle=False,
                                                   num_workers=(0 if opt.test_tta else opt.workers))

        try:
            with open(opt.label_path, encoding='utf-8') as f:
                label = list(map(lambda x: x.strip(), f.readlines()))
        except UnicodeDecodeError:
            with open(opt.label_path, encoding='gbk') as f:
                label = list(map(lambda x: x.strip(), f.readlines()))

        return opt, model, test_dataset, DEVICE, CLASS_NUM, label, save_path


if __name__ == '__main__':
    opt, model, test_dataset, DEVICE, CLASS_NUM, label, save_path = parse_opt()
    y_true, y_pred, y_score, y_feature, img_path = [], [], [], [], []
    with torch.inference_mode():
        for x, y, path in tqdm.tqdm(test_dataset, desc='Test Stage'):
            x = (x.half().to(DEVICE) if opt.half else x.to(DEVICE))
            if opt.test_tta:
                bs, ncrops, c, h, w = x.size()
                pred = model(x.view(-1, c, h, w))
                pred = pred.view(bs, ncrops, -1).mean(1)

                if opt.tsne:
                    pred_feature = model.forward_features(x.view(-1, c, h, w))
                    pred_feature = pred_feature.view(bs, ncrops, -1).mean(1)
            else:
                pred = model(x)

                if opt.tsne:
                    pred_feature = model.forward_features(x)
            try:
                pred = torch.softmax(pred, 1)
            except TypeError:
                pred = torch.softmax(torch.from_numpy(pred), 1)

            y_true.extend(list(y.cpu().detach().numpy()))
            y_pred.extend(list(pred.argmax(-1).cpu().detach().numpy()))
            y_score.extend(list(pred.max(-1)[0].cpu().detach().numpy()))
            img_path.extend(list(path))

            if opt.tsne:
                y_feature.extend(list(pred_feature.cpu().detach().numpy()))

    classification_metric(np.array(y_true), np.array(y_pred), CLASS_NUM, label, save_path)
    if opt.visual:
        visual_predictions(np.array(y_true), np.array(y_pred), np.array(y_score), np.array(img_path), label, save_path)
    if opt.tsne:
        visual_tsne(np.array(y_feature), np.array(y_pred), np.array(img_path), label, save_path)
