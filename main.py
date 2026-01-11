import argparse
import copy
import datetime
import importlib.util
import os
import random
import shutil
import time
import warnings
from copy import deepcopy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
from PIL import ImageFile
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter

from utils import utils_aug
from utils.utils import (
    ModelEMA, WarmUpLR, check_batch_size, de_parallel, dict_to_PrettyTable, freeze_backbone,
    get_channels, load_weights, plot_log, plot_train_batch, save_model, select_device,
    setting_optimizer, show_config, update_opt, init_distributed, cleanup_distributed,
    get_rank, get_world_size, is_main_process, reduce_tensor
)
from utils.utils_distill import *
from utils.utils_fit import fitting, fitting_distill
from utils.utils_loss import *
from utils.utils_model import select_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.deterministic = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet18', help='model name')
    parser.add_argument('--pretrained', action="store_true", help='using pretrain weight')
    parser.add_argument('--weight', type=str, default='', help='loading weight path')
    parser.add_argument('--config', type=str, default='config/config.py', help='config path')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP local rank, -1 for non-DDP')
    parser.add_argument('--sync_bn', action="store_true", help='use SyncBatchNorm for DDP')

    parser.add_argument('--train_path', type=str, default=r'dataset/train', help='train data path')
    parser.add_argument('--val_path', type=str, default=r'dataset/val', help='val data path')
    parser.add_argument('--test_path', type=str, default=r'dataset/test', help='test data path')
    parser.add_argument('--label_path', type=str, default=r'dataset/label.txt', help='label path')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--image_channel', type=int, default=3, help='image channel')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (-1 for autobatch)')
    parser.add_argument('--accumulate', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save path for model and log')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping max norm (0 to disable)')
    parser.add_argument('--resume', action="store_true", help='resume from save_path traning')

    # optimizer parameters
    parser.add_argument('--loss', type=str, choices=['PolyLoss', 'CrossEntropyLoss', 'FocalLoss'],
                        default='CrossEntropyLoss', help='loss function')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW', 'RMSProp'], default='AdamW', help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--class_balance', action="store_true", help='using class balance in loss')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum in optimizer')
    parser.add_argument('--amp', action="store_true", help='using AMP(Automatic Mixed Precision)')
    parser.add_argument('--warmup', action="store_true", help='using WarmUp LR')
    parser.add_argument('--warmup_ratios', type=float, default=0.05,
                        help='warmup_epochs = int(warmup_ratios * epoch) if warmup=True')
    parser.add_argument('--warmup_minlr', type=float, default=1e-6,
                        help='minimum lr in warmup(also as minimum lr in training)')
    parser.add_argument('--metric', type=str, choices=['loss', 'acc', 'mean_acc', 'f1'], default='acc', help='best.pt save rule')
    parser.add_argument('--patience', type=int, default=15, help='EarlyStopping patience (--metric without improvement)')

    # Data Processing parameters
    parser.add_argument('--imagenet_meanstd', action="store_true", help='using ImageNet Mean and Std')
    parser.add_argument('--mixup', type=str, choices=['mixup', 'cutmix', 'none'], default='none', help='MixUp Methods')
    parser.add_argument('--Augment', type=str,
                        choices=['RandAugment', 'AutoAugment', 'TrivialAugmentWide', 'AugMix', 'none'], default='none',
                        help='Data Augment')
    parser.add_argument('--test_tta', action="store_true", help='using TTA')

    # Knowledge Distillation parameters
    parser.add_argument('--kd', action="store_true", help='Knowledge Distillation')
    parser.add_argument('--kd_ratio', type=float, default=0.7, help='Knowledge Distillation Loss ratio')
    parser.add_argument('--kd_method', type=str, choices=['SoftTarget', 'MGD', 'SP', 'AT'], default='SoftTarget', help='Knowledge Distillation Method')
    parser.add_argument('--teacher_path', type=str, default='', help='teacher model path')

    # Tricks parameters
    parser.add_argument('--rdrop', action="store_true", help='using R-Drop')
    parser.add_argument('--ema', action="store_true", help='using EMA(Exponential Moving Average) Reference to YOLOV5')
    parser.add_argument('--freeze_backbone', action="store_true", help='freeze backbone layers, only train classifier')
    parser.add_argument('--freeze_epochs', type=int, default=0, help='freeze backbone for N epochs then unfreeze (0 means freeze all epochs)')

    opt = parser.parse_known_args()[0]
    if opt.resume:
        opt.resume = True
        if not os.path.exists(os.path.join(opt.save_path, 'last.pt')):
            raise FileNotFoundError('last.pt not found. please check your --save_path folder and --resume parameters')
        ckpt = torch.load(os.path.join(opt.save_path, 'last.pt'))
        opt = ckpt['opt']
        opt.resume = True
        print('found checkpoint from {}, model type:{}\n{}'.format(opt.save_path, ckpt['model'].name, dict_to_PrettyTable(ckpt['best_metric'], 'Best Metric')))
    else:
        # Add numeric suffix if folder already exists
        base_path = opt.save_path
        counter = 2
        while os.path.exists(opt.save_path):
            opt.save_path = f"{base_path}{counter}"
            counter += 1
        os.makedirs(opt.save_path)
        spec = importlib.util.spec_from_file_location('config', opt.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.Config()
        shutil.copy(__file__, os.path.join(opt.save_path, 'main.py'))
        shutil.copy(opt.config, os.path.join(opt.save_path, 'config.py'))
        opt = update_opt(opt, config._get_opt())

    set_seed(opt.random_seed + get_rank())  # Use different random seed for each process
    if is_main_process():
        show_config(deepcopy(opt))

    CLASS_NUM = len(os.listdir(opt.train_path))
    
    # Initialize DDP
    is_ddp = opt.local_rank != -1
    if is_ddp:
        init_distributed(opt.local_rank)
        DEVICE = torch.device('cuda', opt.local_rank)
    else:
        DEVICE = select_device(opt.device, opt.batch_size)

    train_transform, test_transform = utils_aug.get_dataprocessing(torchvision.datasets.ImageFolder(opt.train_path),
                                                                   opt)
    train_dataset = torchvision.datasets.ImageFolder(opt.train_path, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(opt.val_path, transform=test_transform)
    if opt.resume:
        model = ckpt['model'].to(DEVICE).float()
    else:
        model = select_model(opt.model_name, CLASS_NUM, (opt.image_size, opt.image_size), opt.image_channel,
                             opt.pretrained)
        model = load_weights(model, opt).to(DEVICE)
        if is_main_process():
            plot_train_batch(copy.deepcopy(train_dataset), opt)
    
    # Freeze backbone
    if opt.freeze_backbone and not opt.resume:
        freeze_backbone(de_parallel(model), freeze=True)

    # DDP SyncBatchNorm
    if is_ddp and opt.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main_process():
            print('Using SyncBatchNorm')

    # Multi-GPU parallel training
    if is_ddp:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=False)
        if is_main_process():
            print(f'Using DistributedDataParallel with {get_world_size()} GPUs')
    else:
        gpu_ids = [int(x) for x in opt.device.split(',') if x.strip().isdigit()]
        if len(gpu_ids) > 1 and torch.cuda.is_available():
            print(f'Using DataParallel with GPUs: {gpu_ids}')
            model = torch.nn.DataParallel(model, device_ids=list(range(len(gpu_ids))))

    batch_size = opt.batch_size if opt.batch_size != -1 else check_batch_size(de_parallel(model), opt.image_size, amp=opt.amp)

    if opt.class_balance:
        class_weight = np.sqrt(compute_class_weight('balanced', classes=np.unique(train_dataset.targets), y=train_dataset.targets))
    else:
        class_weight = np.ones_like(np.unique(train_dataset.targets))
    if is_main_process():
        print('class weight: {}'.format(class_weight))

    # DDP: Use DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_ddp else None
    
    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=is_ddp  # Drop incomplete batch in DDP mode
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, max(batch_size // (10 if opt.test_tta else 1), 1),
        shuffle=False, 
        sampler=test_sampler,
        num_workers=(0 if opt.test_tta else opt.workers),
        pin_memory=True
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(opt.amp if torch.cuda.is_available() else False))
    ema = ModelEMA(model) if opt.ema else None
    optimizer = setting_optimizer(opt, model)
    lr_scheduler = WarmUpLR(optimizer, opt)
    if opt.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        loss = ckpt['loss'].to(DEVICE)
        scaler.load_state_dict(ckpt['scaler'])
        if opt.ema:
            ema.ema = ckpt['ema'].to(DEVICE).float()
            ema.updates = ckpt['updates']
    else:
        loss = eval(opt.loss)(label_smoothing=opt.label_smoothing,
                              weight=torch.from_numpy(class_weight).to(DEVICE).float())
        if opt.rdrop:
            loss = RDropLoss(loss)
    return opt, model, ema, train_dataset_loader, test_dataset_loader, optimizer, scaler, lr_scheduler, loss, DEVICE, CLASS_NUM, (
        ckpt['epoch'] if opt.resume else 0), (ckpt['best_metric'] if opt.resume else None), (train_sampler if is_ddp else None)


if __name__ == '__main__':
    opt, model, ema, train_dataset, test_dataset, optimizer, scaler, lr_scheduler, loss, DEVICE, CLASS_NUM, begin_epoch, best_metric, train_sampler = parse_opt()
    is_ddp = opt.local_rank != -1
    
    if not opt.resume and is_main_process():
        save_epoch = 0
        with open(os.path.join(opt.save_path, 'train.log'), 'w+') as f:
            if opt.kd:
                f.write('epoch,lr,loss,kd_loss,acc,mean_acc,f1,test_loss,test_acc,test_mean_acc,test_f1')
            else:
                f.write('epoch,lr,loss,acc,mean_acc,f1,test_loss,test_acc,test_mean_acc,test_f1')
    elif opt.resume:
        save_epoch = torch.load(os.path.join(opt.save_path, 'last.pt'))['best_epoch']
    else:
        save_epoch = 0

    if opt.kd:
        if not os.path.exists(os.path.join(opt.teacher_path, 'best.pt')):
            raise FileNotFoundError('teacher best.pt not found. please check your --teacher_path folder')
        teacher_ckpt = torch.load(os.path.join(opt.teacher_path, 'best.pt'))
        teacher_model = teacher_ckpt['model'].float().to(DEVICE).eval()
        if is_main_process():
            print('found teacher checkpoint from {}, model type:{}\n{}'.format(opt.teacher_path, teacher_model.name, dict_to_PrettyTable(teacher_ckpt['best_metric'], 'Best Metric')))
        
        if opt.resume:
            kd_loss = torch.load(os.path.join(opt.save_path, 'last.pt'))['kd_loss'].to(DEVICE)
        else:
            if opt.kd_method == 'SoftTarget':
                kd_loss = SoftTarget().to(DEVICE)
            elif opt.kd_method == 'MGD':
                kd_loss = MGD(get_channels(model, opt), get_channels(teacher_model, opt)).to(DEVICE)
                optimizer.add_param_group({'params': kd_loss.parameters(), 'weight_decay': opt.weight_decay})
            elif opt.kd_method == 'SP':
                kd_loss = SP().to(DEVICE)
            elif opt.kd_method == 'AT':
                kd_loss = AT().to(DEVICE)

    # Initialize TensorBoard (only on main process)
    writer = SummaryWriter(log_dir=os.path.join(opt.save_path, 'tensorboard')) if is_main_process() else None

    if is_main_process():
        print('{} begin train!'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    for epoch in range(begin_epoch, opt.epoch):
        if epoch > (save_epoch + opt.patience) and opt.patience != 0:
            if is_main_process():
                print('No Improve from {} to {}, EarlyStopping.'.format(save_epoch + 1, epoch))
            break

        # DDP: Set epoch to ensure different shuffle for each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Unfreeze backbone after specified epochs
        if opt.freeze_backbone and opt.freeze_epochs > 0 and epoch == opt.freeze_epochs:
            if is_main_process():
                print(f'Epoch {epoch + 1}: Unfreezing backbone...')
            freeze_backbone(de_parallel(model), freeze=False)
            # Reset optimizer to include unfrozen parameters
            optimizer = setting_optimizer(opt, de_parallel(model))

        begin = time.time()
        if opt.kd:
            metric = fitting_distill(teacher_model, model, ema, loss, kd_loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler, '{}/{}'.format(epoch + 1,opt.epoch), opt, is_ddp)
        else:
            metric = fitting(model, ema, loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler,'{}/{}'.format(epoch + 1, opt.epoch), opt, is_ddp)
        
        # Only log on main process
        if is_main_process():
            with open(os.path.join(opt.save_path, 'train.log'), 'a+') as f:
                f.write(
                    '\n{},{:.10f},{}'.format(epoch + 1, optimizer.param_groups[2]['lr'], metric[1]))

        n_lr = optimizer.param_groups[2]['lr']
        
        # TensorBoard logging (only on main process)
        if is_main_process() and writer is not None:
            writer.add_scalar('Loss/train', metric[0]['train_loss'], epoch + 1)
            writer.add_scalar('Loss/val', metric[0]['test_loss'], epoch + 1)
            writer.add_scalar('Accuracy/train', metric[0]['train_acc'], epoch + 1)
            writer.add_scalar('Accuracy/val', metric[0]['test_acc'], epoch + 1)
            writer.add_scalar('Mean_Accuracy/train', metric[0]['train_mean_acc'], epoch + 1)
            writer.add_scalar('Mean_Accuracy/val', metric[0]['test_mean_acc'], epoch + 1)
            writer.add_scalar('F1/train', metric[0]['train_f1'], epoch + 1)
            writer.add_scalar('F1/val', metric[0]['test_f1'], epoch + 1)
            writer.add_scalar('Learning_Rate', n_lr, epoch + 1)
            if opt.kd and 'train_kd_loss' in metric[0]:
                writer.add_scalar('Loss/kd', metric[0]['train_kd_loss'], epoch + 1)
        
        lr_scheduler.step()

        # Only save model on main process
        if is_main_process():
            if best_metric is None:
                best_metric = metric[0]
            else:
                if eval('{} {} {}'.format(metric[0]['test_{}'.format(opt.metric)], '<' if opt.metric == 'loss' else '>', best_metric['test_{}'.format(opt.metric)])):
                    best_metric = metric[0]
                    save_model(
                        os.path.join(opt.save_path, 'best.pt'),
                        **{
                        'model': (deepcopy(ema.ema).to('cpu').half() if opt.ema else deepcopy(de_parallel(model)).to('cpu').half()),
                        'opt': opt,
                        'best_metric': best_metric,
                        }
                    )
                    save_epoch = epoch
            
            save_model(
                os.path.join(opt.save_path, 'last.pt'),
                **{
                   'model': deepcopy(de_parallel(model)).to('cpu').half(),
                   'ema': (deepcopy(ema.ema).to('cpu').half() if opt.ema else None),
                   'updates': (ema.updates if opt.ema else None),
                   'opt': opt,
                   'epoch': epoch + 1,
                   'optimizer': optimizer.state_dict(),
                   'lr_scheduler': lr_scheduler.state_dict(),
                   'best_metric': best_metric,
                   'loss': deepcopy(loss).to('cpu'),
                   'kd_loss': (deepcopy(kd_loss).to('cpu') if opt.kd else None),
                   'scaler': scaler.state_dict(),
                   'best_epoch': save_epoch,
                }
            )

            print(dict_to_PrettyTable(metric[0], '{} epoch:{}/{}, best_epoch:{}, time:{:.2f}s, lr:{:.8f}'.format(
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch + 1, opt.epoch, save_epoch + 1, time.time() - begin, n_lr,
                    )))
        
        # DDP synchronization
        if is_ddp:
            dist.barrier()
    
    # Close TensorBoard
    if is_main_process() and writer is not None:
        writer.close()
    
    # Cleanup DDP
    if is_ddp:
        cleanup_distributed()
    
    if is_main_process():
        plot_log(opt)
