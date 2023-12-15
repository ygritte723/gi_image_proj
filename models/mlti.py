import numpy as np
import random
import torch
from torch.distributions import Beta


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam.cpu())
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    # uniformly select center point coordinates
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # calculate bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_data(xs, xq, lam):
    mixed_x = xq.clone()
    bbx1, bby1, bbx2, bby2 = rand_bbox(xq.size(), lam)
    # cut and paste source set data into result set data at the selected bounding box
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = xs[:, :, bbx1:bbx2, bby1:bby2]
    return mixed_x


def cross_mix(data, data_aux, data_sec, data_aux_sec):
    lam_mix = Beta(torch.FloatTensor([2]), torch.FloatTensor([2])).sample().to("cuda")
    # apply augmentation on both the data and the auxiliary data
    data = mixup_data(data, data_sec, lam_mix)
    data_aux = mixup_data(data_aux, data_aux_sec, lam_mix)
    return data, data_aux
