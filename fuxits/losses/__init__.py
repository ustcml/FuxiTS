import math, torch
from torch.nn.modules.loss import L1Loss, MSELoss, HuberLoss, SmoothL1Loss
from losses import MAPELoss, MaskedLoss

def ismask(data, mask_val):
    assert mask_val is not None
    mask = ~torch.isnan(data)
    if not math.isnan(mask_val):
        mask = mask & (torch.abs(data - mask_val) > torch.finfo('float').eps)
    return mask
    