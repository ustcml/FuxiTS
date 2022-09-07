import torch.nn as nn
from torch import Tensor
from . import ismask
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch

def divide_no_nan(a: Tensor, b: Tensor):
    div = a/b
    return torch.nan_to_num(div, 0., 0., 0.)


class MaskedLoss(nn.Module):
    '''the parameter ``reduction'' of loss should be set to ``none"
    '''
    def __init__(self, loss: _Loss, missing_val=float('nan')) -> None:
        assert missing_val != None
        assert loss.reduction != 'none'
        super().__init__()
        self.loss = loss
        self.missing_value = missing_val

    def forward(self, input: Tensor, target: Tensor):
        mask = ismask(target, self.missing_value).float()
        mask_loss = self.loss(input, target) * mask / mask.mean()
        return mask_loss.mean()


class MAPELoss(_Loss):

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MAPELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = divide_no_nan(input, target)
        input = torch.ones_like(input)
        return F.l1_loss(input, target, reduction=self.reduction)
