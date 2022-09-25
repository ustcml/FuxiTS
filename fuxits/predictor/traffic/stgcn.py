from ctypes import Union
from turtle import forward
from fuxits.predictor.predictor import Predictor
import torch.nn as nn
import torch
import torch.nn.functional as F

class STGCN(Predictor):
    
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)



class FeatAlign(nn.Module):
    def __init__(self, c_in, c_out):
        super(FeatAlign, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor):
        # x should be B x C x T x N
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in])
        return x



class Temporal_Conv_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, time_cov_kernel, activation="relu"):
        super(Temporal_Conv_Layer, self).__init__()
        assert activation in {'relu', 'sigmoid', 'glu'}
        self.time_cov_kernel = time_cov_kernel
        self.activation = activation
        self.feat_align = FeatAlign(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels * (2 if activation == "glu" else 1), (time_cov_kernel, 1), 1)

    def forward(self, x):
        # x should be B x C x T x N
        x_align = self.feat_align(x)
        x_align = x_align[:, :, self.time_cov_kernel - 1:, :]
        if self.activation == "glu":
            output_x, output_sigma = torch.chunk(self.conv(x), 2)
            return (output_x + x_align) * torch.sigmoid(output_sigma)
        elif self.activation == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_align)
        elif self.activation == 'relu':
            return torch.relu(self.conv(x) + x_align)


class Spatial_Conv_Layer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Spatial_Conv_Layer).__init__()
    
    def forward(self, x: torch.Tensor):
        
        pass