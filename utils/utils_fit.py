from copy import deepcopy

import numpy as np
import torch
import tqdm

from .utils import TrainMetric
from .utils_aug import mixup_criterion, mixup_data

def fitting(model, ema, loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler, show_thing, opt):
    model.train()
    metric = TrainMetric(CLASS_NUM)
    accumulate = getattr(opt, 'accumulate', 1)
    grad_clip = getattr(opt, 'grad_clip', 0.0)
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_dataset, desc='{} Train Stage'.format(show_thing))):
        x, y = x.to(DEVICE).float(), y.to(DEVICE).long()

        with torch.cuda.amp.autocast(opt.amp):
            if opt.rdrop:
                if opt.mixup != 'none' and np.random.rand() > 0.5:
                    x_mixup, y_a, y_b, lam = mixup_data(x, y, opt)
                    pred = model(x_mixup)
                    pred2 = model(x_mixup)
                    l = mixup_criterion(loss, [pred, pred2], y_a, y_b, lam)
                    pred = model(x)
                else:
                    pred = model(x)
                    pred2 = model(x)
                    l = loss([pred, pred2], y)
            else:
                if opt.mixup != 'none' and np.random.rand() > 0.5:
                    x_mixup, y_a, y_b, lam = mixup_data(x, y, opt)
                    pred = model(x_mixup)
                    l = mixup_criterion(loss, pred, y_a, y_b, lam)
                    pred = model(x)
                else:
                    pred = model(x)
                    l = loss(pred, y)

        # Gradient accumulation: divide loss by accumulation steps
        l = l / accumulate
        metric.update_loss(float(l.data) * accumulate)
        metric.update_y(y, pred)
        
        scaler.scale(l).backward()

        # Update parameters every accumulate steps
        if (batch_idx + 1) % accumulate == 0 or (batch_idx + 1) == len(train_dataset):
            # Gradient clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

    if ema:
        model_eval = ema.ema
    else:
        model_eval = model.eval()
    with torch.inference_mode():
        for x, y in tqdm.tqdm(test_dataset, desc='{} Test Stage'.format(show_thing)):
            x, y = x.to(DEVICE).float(), y.to(DEVICE).long()

            with torch.cuda.amp.autocast(opt.amp):
                if opt.test_tta:
                    bs, ncrops, c, h, w = x.size()
                    pred = model_eval(x.view(-1, c, h, w))
                    pred = pred.view(bs, ncrops, -1).mean(1)
                    l = loss(pred, y)
                else:
                    pred = model_eval(x)
                    l = loss(pred, y)
                
            metric.update_loss(float(l.data), isTest=True)
            metric.update_y(y, pred, isTest=True)

    return metric.get()


def fitting_distill(teacher_model, student_model, ema, loss, kd_loss, optimizer, train_dataset, test_dataset, CLASS_NUM,
                    DEVICE, scaler, show_thing, opt):
    student_model.train()
    metric = TrainMetric(CLASS_NUM)
    accumulate = getattr(opt, 'accumulate', 1)
    grad_clip = getattr(opt, 'grad_clip', 0.0)
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_dataset, desc='{} Train Stage'.format(show_thing))):
        x, y = x.to(DEVICE).float(), y.to(DEVICE).long()

        with torch.cuda.amp.autocast(opt.amp):
            if opt.mixup != 'none' and np.random.rand() > 0.5:
                x_mixup, y_a, y_b, lam = mixup_data(x, y, opt)
                s_features, s_features_fc, s_pred = student_model(x_mixup, need_fea=True)
                t_features, t_features_fc, t_pred = teacher_model(x_mixup, need_fea=True)
                l = mixup_criterion(loss, s_pred, y_a, y_b, lam)
                pred = student_model(x)
            else:
                s_features, s_features_fc, s_pred = student_model(x, need_fea=True)
                t_features, t_features_fc, t_pred = teacher_model(x, need_fea=True)
                l = loss(s_pred, y)
            if str(kd_loss) in ['SoftTarget']:
                kd_l = kd_loss(s_pred, t_pred)
            elif str(kd_loss) in ['MGD']:
                kd_l = kd_loss(s_features[-1], t_features[-1])
            elif str(kd_loss) in ['SP']:
                kd_l = kd_loss(s_features[2], t_features[2]) + kd_loss(s_features[3], t_features[3])
            elif str(kd_loss) in ['AT']:
                kd_l = kd_loss(s_features[2], t_features[2]) + kd_loss(s_features[3], t_features[3])
                    
            if str(kd_loss) in ['SoftTarget', 'SP', 'MGD']:
                kd_l *= (opt.kd_ratio / (1 - opt.kd_ratio)) if opt.kd_ratio < 1 else opt.kd_ratio
            elif str(kd_loss) in ['AT']:
                kd_l *= opt.kd_ratio

        # Gradient accumulation: divide loss by accumulation steps
        total_loss = (l + kd_l) / accumulate
        metric.update_loss(float(l.data))
        metric.update_loss(float(kd_l.data), isKd=True)
        if opt.mixup != 'none':
            metric.update_y(y, pred)
        else:
            metric.update_y(y, s_pred)

        scaler.scale(total_loss).backward()

        # Update parameters every accumulate steps
        if (batch_idx + 1) % accumulate == 0 or (batch_idx + 1) == len(train_dataset):
            # Gradient clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(student_model)

    if ema:
        model_eval = ema.ema
    else:
        model_eval = student_model.eval()
    with torch.inference_mode():
        for x, y in tqdm.tqdm(test_dataset, desc='{} Test Stage'.format(show_thing)):
            x, y = x.to(DEVICE).float(), y.to(DEVICE).long()

            with torch.cuda.amp.autocast(opt.amp):
                if opt.test_tta:
                    bs, ncrops, c, h, w = x.size()
                    pred = model_eval(x.view(-1, c, h, w))
                    pred = pred.view(bs, ncrops, -1).mean(1)
                    l = loss(pred, y)
                else:
                    pred = model_eval(x)
                    l = loss(pred, y)

            metric.update_loss(float(l.data), isTest=True)
            metric.update_y(y, pred, isTest=True)

    return metric.get()
