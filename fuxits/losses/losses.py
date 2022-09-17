import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch, math

def divide_no_nan(a: Tensor, b: Tensor):
    div = a/b
    return torch.nan_to_num(div, 0., 0., 0.)


class MaskedLoss(nn.Module):
    '''the parameter ``reduction'' of loss should be set to ``none"
    '''
    def __init__(self, loss: _Loss, missing_val=float('nan')) -> None:
        if loss.reduction == 'none':
            assert missing_val != None
        super().__init__()
        self.loss = loss
        self.missing_value = missing_val

    def forward(self, input: Tensor, target: Tensor):
        if self.loss.reduction == 'none':
            mask = ismask(target, self.missing_value).float()
            mask_loss = self.loss(input, target) * mask / mask.mean()
            return mask_loss.mean()
        else:
            return self.loss(input, target)


class MAPELoss(_Loss):

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MAPELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = divide_no_nan(input, target)
        input = torch.ones_like(input)
        return F.l1_loss(input, target, reduction=self.reduction)


def ismask(data, mask_val):
    assert mask_val is not None
    mask = ~torch.isnan(data)
    if not math.isnan(mask_val):
        mask = mask & (torch.abs(data - mask_val) > torch.finfo('float').eps)
    return mask