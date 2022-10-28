from ctypes import Union
from fuxits.predictor.predictor import Predictor
import torch.nn as nn
import torch
import torch.nn.functional as F
from fuxits.layers.graph import ChebConv, compute_cheb_poly

class STGCN(Predictor):
    def __init__(self, config, in_channels, num_nodes, hist_steps, pred_steps, static_adj):
        super(STGCN, self).__init__(config)
        cheb_k = self.model_config['cheb_k']
        time_cov_kernel = self.model_config['time_cov_kernel']
        hidden_size = [in_channels] + self.model_config['hidden_size']
        drop_ratio = self.model_config['drop_ratio']
        chebpoly = compute_cheb_poly(static_adj, cheb_k, 'sys')
        self.st_conv1 = STConv(cheb_k, time_cov_kernel, hidden_size[:3], num_nodes, chebpoly, drop_ratio)
        self.st_conv2 = STConv(cheb_k, time_cov_kernel, hidden_size[2:], num_nodes, chebpoly, drop_ratio)
        #self.output = Output(hidden_size[-1], hist_steps - 4 * (time_cov_kernel - 1), num_nodes, pred_steps)
        output_len = hist_steps - 4 * (time_cov_kernel - 1)
        output_kernel = output_len + 1 - pred_steps
        assert output_kernel > 0
        self.output = Output(hidden_size[-1], output_kernel, num_nodes, in_channels)

        
    def forward(self, x, y=None, batch_idx=None):
        x_st1 = self.st_conv1(x)
        x_st2 = self.st_conv2(x_st1)
        return self.output(x_st2)

class FeatAlign(nn.Module):
    def __init__(self, c_in, c_out):
        super(FeatAlign, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor):
        """ Align feature maps into the same space
        Args:
            x: (Tensor, BTNI)
        Return:
            Tensor, BTNI
        """
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in])
        return x

class Output(nn.Module):
    def __init__(self, in_channels, kernels, num_nodes, out_channels) -> None:
        super(Output, self).__init__()
        self.tconv1 = Temporal_Conv(in_channels, in_channels, kernels, "glu")
        self.ln = nn.LayerNorm([num_nodes, in_channels])
        self.tconv2 = Temporal_Conv(in_channels, in_channels, 1, "sigmoid")
        self.fc = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1) 
        x_t2 = self.tconv2(x_ln)
        out = self.fc(x_t2)
        return out



class Temporal_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, time_cov_kernel, activation="relu"):
        super(Temporal_Conv, self).__init__()
        assert activation in {'relu', 'sigmoid', 'glu'}
        self.time_cov_kernel = time_cov_kernel
        self.activation = activation
        self.feat_align = FeatAlign(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels * (2 if activation == "glu" else 1), (1, time_cov_kernel), 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (Tensor, [BTNI])
        Returns:
            Tensor, [BTNI]
        """
        x = x.transpose(1, -1) # [BTNI] ->[BINT]
        x_align = self.feat_align(x)
        x_align = x_align[:, :, :, self.time_cov_kernel - 1:]
        output = None
        if self.activation == "glu":
            output_x, output_sigma = torch.chunk(self.conv(x), 2, dim=1)
            output = (output_x + x_align) * torch.sigmoid(output_sigma)
        elif self.activation == "sigmoid":
            output = torch.sigmoid(self.conv(x) + x_align)
        elif self.activation == 'relu':
            output = torch.relu(self.conv(x) + x_align)
        return output.transpose(1, -1) # [BINT]->[BTNI]


class STConv(nn.Module):
    def __init__(self, cheb_k, time_cov_kernel, hidden_size, num_nodes, static_adj, p, **kwargs) -> None:
        super(STConv, self).__init__()
        self.tconv1 = Temporal_Conv(hidden_size[0], hidden_size[1], time_cov_kernel, 'glu')
        #self.sconv = ChebConv(hidden_size[1], hidden_size[1], cheb_k, static_adj, bias=True)
        self.sconv = ChebConv(hidden_size[1], hidden_size[1], 1, cheb_k, bias=True)
        self.register_buffer('static_adj', static_adj)
        self.tconv2 = Temporal_Conv(hidden_size[1], hidden_size[2], time_cov_kernel)
        self.ln = nn.LayerNorm([num_nodes, hidden_size[2]])
        self.dropout = nn.Dropout(p)
    def forward(self, x):
        #x of size b-f-n-t
        x_t1 = self.tconv1(x) #[BTNI]->[BSNO]
        x_s = F.relu(self.sconv(x_t1, self.static_adj) + x_t1) # [BTNI]->[BTNO]
        x_t2 = self.tconv2(x_s) # [BTNI]->[BSNO]
        x_ln = self.ln(x_t2)     #bfnt->btnf 
        return self.dropout(x_ln)
       



