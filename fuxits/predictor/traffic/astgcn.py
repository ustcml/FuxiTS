from fuxits import losses
from fuxits.predictor.predictor import Predictor
import torch.nn as nn
import torch
import torch.nn.functional as F
from fuxits.predictor.graphutils import laplacian, scale_lapacian
class ASTGCN(Predictor):
    def __init__(self, config, in_channels, num_nodes, hist_steps, pred_steps, static_adj, **kwargs):
        super().__init__(config)
        cheb_k = self.model_config['cheb_k']
        time_cov_strides = self.model_config['time_cov_strides']
        num_chev_filter = self.model_config['num_chev_filter']
        num_time_filter = self.model_config['num_time_filter']
        chebpoly = torch.stack(list(compute_cheb_poly(static_adj, cheb_k)))
        self.submodules = nn.ModuleList([ASTGCN_Sub(num_nodes, _, in_channels, chebpoly, pred_steps, num_chev_filter, cheb_k, num_time_filter, time_cov_strides) for _ in hist_steps])
        if len(self.submodules) > 1:
            self.weight = nn.parameter.Parameter(torch.empty(self.train_config['batch_size'], num_nodes, pred_steps))
        self.reset_parameters()
    
    def forward(self, x):
        if len(self.submodules) > 1:
            return torch.stack([m(_.permute(0, 2, 3, 1)) * w for m, w, _ in zip(self.submodules, self.weight, x)]).sum(dim=0)
        else:
            return self.submodules[0](x[0].permute(0, 2, 3, 1))
    
    def _get_loss(self):
        return losses.L1Loss()

        

class ASTGCN_Sub(nn.Module):
    def __init__(self, num_nodes, hist_steps, in_channels, static_adj, pred_steps, num_chev_filter, cheb_k, num_time_filter, time_cov_strides, num_blocks=2) -> None:
        super().__init__()
        self.modulelist = nn.ModuleList([ASTGCN_Block(num_nodes, hist_steps, in_channels, num_chev_filter, cheb_k, num_time_filter, time_cov_strides, static_adj)] + \
            [ASTGCN_Block(num_nodes, hist_steps // time_cov_strides, num_time_filter, num_chev_filter, cheb_k, num_time_filter, 1, static_adj) for _ in range(num_blocks - 1)])
        self.conv = nn.Conv2d(hist_steps // time_cov_strides, pred_steps, kernel_size=(1, num_time_filter))

    def forward(self, x):
        for m in self.modulelist:
            x = m(x)
        return self.conv(x.permute(0, 3, 1, 2))

class ASTGCN_Block(nn.Module):
    def __init__(self, num_nodes, hist_steps, in_channels, num_chev_filter, cheb_k, num_time_filter, time_cov_strides, static_adj) -> None:
        super(ASTGCN_Block, self).__init__()
        self.TAt = Temporal_Attention_layer(in_channels, num_nodes, hist_steps)
        self.SAt = Spatial_Attention_layer(in_channels, num_nodes, hist_steps)
        self.cheb_conv = ChebConv(in_channels, num_chev_filter, cheb_k, static_adj)
        self.time_conv = nn.Conv2d(num_chev_filter, num_time_filter, kernel_size=(1, 3), stride=(1, time_cov_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, num_time_filter, kernel_size=(1, 1), stride=(1, time_cov_strides))
        self.ln = nn.LayerNorm(num_time_filter)  #需要将channel放到最后一个维度上
       

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        x1 = self.TAt(x)
        x2 = self.cheb_conv(x1, self.SAt(x1))
        x3 = self.time_conv(x2.transpose(1, 2))
        x_residual = self.residual_conv(x.transpose(1, 2))
        x_residual = self.ln(F.relu(x_residual + x3).transpose(1, 3)).permute(0, 2, 3, 1)
        return x_residual


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, in_channels, num_nodes, num_steps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_steps))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_steps))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.Vs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, N, N)
        '''
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = F.softmax(S, dim=1)
        return S_normalized


class Temporal_Attention_layer(nn.Module):
    def __init__(self, in_channels, num_nodes, num_steps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_nodes))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_nodes))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, num_steps, num_steps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_steps, num_steps))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        lhs = torch.matmul(torch.matmul(x.transpose(1, 3), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)
        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        return torch.matmul(x.view(x.shape[0], -1, x.shape[-1]), F.softmax(E, dim=1)).view(*x.shape)

class ChebConv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, in_channels, out_channels, K, static_adj):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(ChebConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)
        ])
        if static_adj.dim() == 3:
            self.register_buffer('chebpoly', static_adj)
            #self.chebpoly = static_adj
        else:
            #self.chebpoly = torch.stack(list(compute_cheb_poly(static_adj, K)))
            self.register_buffer('chebpoly', torch.stack(list(compute_cheb_poly(static_adj, K))))
            

    def forward(self, x, dyna_adj):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
        output = torch.zeros(*x.shape[:2], self.out_channels, x.shape[-1]).type_as(x)
        for k in range(self.chebpoly.shape[0]):
            T_k = self.chebpoly[k] * dyna_adj # (B, N, N)
            rhs = torch.matmul(x.view(x.shape[0], x.shape[1], -1).transpose(1, 2), T_k).transpose(1, 2).view(*x.shape) # (B, N, F, T)
            output = output + self.lins[k](rhs.transpose(-1, -2)).transpose(-1, -2)
        return F.relu(output)



def compute_cheb_poly(adj, K):
        '''
        compute a list of chebyshev polynomials from T_0 to T_{K-1}
        Parameters
        ----------
        L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
        K: the maximum order of chebyshev polynomials
        Returns
        ----------
        cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
        '''
        adj = torch.from_numpy(adj).type(torch.float32)
        adj[adj > 1e-10] = 1
        L_tilde = scale_lapacian(laplacian(adj))
        L0 = torch.eye(L_tilde.shape[0])
        yield L0
        L1 = L_tilde
        yield L1
        for _ in range(2, K):
            L2 = 2 * L_tilde * L1 - L0
            L0 = L1
            L1 = L2
            yield L2