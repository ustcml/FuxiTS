from fuxits import losses
from fuxits.predictor.predictor import Predictor
import torch.nn as nn
import torch
import torch.nn.functional as F
from fuxits.layers.graph import compute_cheb_poly, ChebConv

class ASTGCN(Predictor):
    def __init__(self, config, in_channels, num_nodes, hist_steps, pred_steps, static_adj, **kwargs):
        super().__init__(config)
        cheb_k = self.model_config['cheb_k']
        time_cov_strides = self.model_config['time_cov_strides']
        num_chev_filter = self.model_config['num_chev_filter']
        num_time_filter = self.model_config['num_time_filter']
        chebpoly = compute_cheb_poly(static_adj, cheb_k)
        if isinstance(hist_steps, list):
            self.submodules = nn.ModuleList([ASTGCN_Sub(num_nodes, _, in_channels, chebpoly, pred_steps, num_chev_filter, cheb_k, num_time_filter, time_cov_strides) for _ in hist_steps])
            if len(self.submodules) > 1:
                self.weight = nn.parameter.Parameter(torch.FloatTensor(1, in_channels, num_nodes, pred_steps))
        else:
            self.submodules = ASTGCN_Sub(num_nodes, hist_steps, in_channels, chebpoly, pred_steps, num_chev_filter, cheb_k, num_time_filter, time_cov_strides)
    
    def forward(self, x):
        if isinstance(self.submodules, nn.ModuleList):
            if len(self.submodules) > 1:
                return torch.stack([m(_) * w for m, w, _ in zip(self.submodules, self.weight, x)]).sum(dim=0)
            else:
                return self.submodules[0](x[0])
        else:
            return self.submodules(x)
    
    def _get_loss(self):
        return losses.L1Loss()

        

class ASTGCN_Sub(nn.Module):
    def __init__(self, num_nodes, hist_steps, in_channels, static_adj, pred_steps, num_chev_filter, cheb_k, num_time_filter, time_cov_strides, num_blocks=2) -> None:
        super().__init__()
        self.modulelist = nn.ModuleList([ASTGCN_Block(num_nodes, hist_steps, in_channels, num_chev_filter, cheb_k, num_time_filter, time_cov_strides, static_adj)] + \
            [ASTGCN_Block(num_nodes, hist_steps // time_cov_strides, num_time_filter, num_chev_filter, cheb_k, num_time_filter, 1, static_adj) for _ in range(num_blocks - 1)])
        self.conv = nn.Conv2d(hist_steps // time_cov_strides, pred_steps, kernel_size=(1, num_time_filter - in_channels + 1))

    def forward(self, x):
        for m in self.modulelist:
            x = m(x)
        return self.conv(x) # [BTNF]->[B Tout N Fin]

class ASTGCN_Block(nn.Module):
    def __init__(self, num_nodes, hist_steps, in_channels, num_chev_filter, cheb_k, num_time_filter, time_cov_strides, static_adj) -> None:
        super(ASTGCN_Block, self).__init__()
        self.TAt = Temporal_Attention(in_channels, num_nodes, hist_steps)
        self.SAt = Spatial_Attention(in_channels, num_nodes, hist_steps)
        self.cheb_conv = ChebConv(in_channels, num_chev_filter, 1, cheb_k)
        self.register_buffer('static_adj', static_adj)
        self.time_conv = nn.Conv2d(num_chev_filter, num_time_filter, kernel_size=(1, 3), stride=(1, time_cov_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, num_time_filter, kernel_size=(1, 1), stride=(1, time_cov_strides))
        self.ln = nn.LayerNorm(num_time_filter)  #需要将channel放到最后一个维度上
       

    def forward(self, x):
        '''
        :param x: (B, T, N, F_in)
        :return: (B, T, N, F_out)
        '''
        x1 = self.TAt(x)
        adj = torch.unsqueeze(self.static_adj, 1) * torch.unsqueeze(self.SAt(x1), 0)
        x2 = F.relu(self.cheb_conv(x, adj))
        x3 = self.time_conv(x2.transpose(1, 3))
        x_residual = self.residual_conv(x.transpose(1, 3))
        x_residual = self.ln(F.relu(x_residual + x3).transpose(1, 3))
        return x_residual


class Spatial_Attention(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, in_channels, num_nodes, num_steps):
        super(Spatial_Attention, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_steps))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_steps))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.Vs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W2.T)
        nn.init.xavier_uniform_(self.Vs)
        #nn.init.xavier_uniform_(self.W1.view(1, -1))
        #nn.init.xavier_uniform_(self.W3.view(1, -1))
        nn.init.uniform_(self.W1)
        nn.init.uniform_(self.W3)
        nn.init.uniform_(self.bs)
        

    def forward(self, x):
        '''
        :param x: (B, T, N, F)
        :return: (B, N, N)
        '''
        lhs = torch.tensordot(x, self.W1, dims=([1], [0])) @ self.W2 #[BTNF][T]->[BNF][FT]->[BNT]
        #torch.matmul(torch.matmul(x, self.W1).transpose(-1, -2), self.W2)  # (b,F,N,T)(T)->(b,N,F)(F,T)->(b,N,T)
        #rhs = torch.sum(self.W3.view(1,-1,1,1) * x, 1).transpose(-1, -2)  # (F)(b,F,N,T)->(b,N,T)->(b,T,N)
        rhs = x @ self.W3 #[BTNF][F]->[BTN]
        product = torch.matmul(lhs, rhs)  # [BNT][BTN]->[BNN]
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = F.softmax(S, dim=1)
        return S_normalized


class Temporal_Attention(nn.Module):
    def __init__(self, in_channels, num_nodes, num_steps):
        super(Temporal_Attention, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_nodes))
        self.U2 = nn.Parameter(torch.FloatTensor(num_nodes, in_channels))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(num_steps, num_steps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_steps, num_steps))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U2)
        nn.init.xavier_uniform_(self.Ve)
        #nn.init.xavier_uniform_(self.U1.view(1, -1))
        #nn.init.xavier_uniform_(self.U3.view(1, -1))
        nn.init.uniform_(self.U1)
        nn.init.uniform_(self.U3)
        nn.init.uniform_(self.be)

    def forward(self, x):
        '''
        :param x: (B, T, N, F)
        :return: (B, T, N, F)
        '''
        #lhs = torch.matmul(self.U2, torch.matmul(x.transpose(-1, -2), self.U1))
        lhs = torch.matmul(torch.tensordot(x, self.U1, dims=([-2],[0])), self.U2.T) # [BTNF][N] -> [BTF], [BTF][FN]->[BTN]
        rhs = x @ self.U3  # [BTNF][F]->[BTN]
        product = torch.matmul(lhs, rhs.transpose(-1, -2))  # [BTN][BNT]->[BTT]
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        #return torch.matmul(x.reshape(x.shape[0], -1, x.shape[-1]), F.softmax(E, dim=1)).view(*x.shape) # [B FN T] [B T T]
        return torch.einsum("btnf,bts->bsnf", x, F.softmax(E, dim=1))




