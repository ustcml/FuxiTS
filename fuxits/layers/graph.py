import torch
from scipy.sparse.linalg import eigs, eigsh
import torch.nn as nn
import numpy as np

def preprocess_graph(mode:str, adj, **kwargs):

    def symmetrize(adj, agg):
        if agg is None:
            return adj
        assert agg in {'max', 'mean', 'min'}
        if agg == 'max':
            #adj_output = np.maximum(adj, adj.T)
            adj_output = np.where(((adj >= adj.T) & ~np.isposinf(adj))|np.isposinf(adj.T), adj, adj.T)
        elif agg == 'mean':
            adj_output = adj + adj.T
            adj_output[(adj > 0) & (adj.T > 0)] /= 2
        elif agg == 'min':
            adj_output = np.minimum(adj, adj.T)
        return adj_output

    def binary_graph(adj, epsilon=np.finfo(float).eps):
        ones_idx = (adj>epsilon*10000) & ~np.isposinf(adj)
        adj[ones_idx] = 1
        adj[~ones_idx] = 0
        return adj

    def epsilon_graph(adj, sigma2=0.1, epsilon=0.5):
        idx = ~np.isposinf(adj)
        if isinstance(sigma2, float):
            output = np.exp(- (adj/10000)**2 / sigma2) 
        elif sigma2 == 'std':
            std = adj[idx].std()
            output = np.exp(- (adj/std)**2)
        output[output < epsilon] = 0.
        np.fill_diagonal(output, 0.)
        return output

    adj = symmetrize(adj, kwargs.pop('sysmetrize_aggregation', None))
    process_funs = {'binary':binary_graph, 'epsilon':epsilon_graph}
    return process_funs[mode](adj, **kwargs)

def graph_adj_norm(adj_t, issys=True, add_self_loops=False):
    if add_self_loops:
        adj_t = add_self_loops_fun(adj_t)
    deg = degree(adj_t)
    if issys:
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
        adj_t = adj_t * deg_inv_sqrt.view(1, -1)
    else:
        deg_inv = deg.pow_(-1)
        adj_t = adj_t * deg_inv.view(-1, 1)
    return adj_t

def degree(adj_t):
    return torch.sum(adj_t, dim=1)

def add_self_loops_fun(adj_t, weight=1.0):
    return adj_t.fill_diagonal_(weight) 

def laplacian(adj_t, normalized=None, add_self_loops=False):
    '''
    compute `\\tilde{L}`
    Parameters
    ----------
    W: torch.tensor, shape is (N, N), N is the num of vertices
    Returns
    normalized: 'sys' and 'rw'
    ----------
    laplacian: torch.tensor, shape (N, N)
    '''

    if normalized:
        W = adj_t
        assert normalized in {'sys', 'rw'}
        W = graph_adj_norm(W, normalized=='sys', add_self_loops)
        return torch.eye(W.shape[0]) - W
    else:
        if add_self_loops:
            adj_t = add_self_loops_fun(adj_t)
        L = torch.diag(degree(adj_t)) - adj_t
    return L

def scale_lapacian(L:torch.Tensor, lambda_max=None):
    if lambda_max is None:
        lambda_max = eigs(L.cpu().numpy(), k=1, which='LR')[0].real
        return (2 * L) / lambda_max - torch.eye(L.shape[0])
    else:
        return (2 * L) / lambda_max - torch.eye(L.shape[0])


def compute_cheb_poly(adj, K, normalized=None, add_self_loop=False):
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
        #adj[adj > 1e-10] = 1 # this should be done outside of this function
        if normalized == 'rw':
            L_tilde = graph_adj_norm(adj, False, add_self_loop)
        else:    
            L_tilde = scale_lapacian(laplacian(adj, normalized='sys', add_self_loops=add_self_loop))
        if K == 1:
            return L_tilde
        else:
            LL = [torch.eye(L_tilde.shape[0]), L_tilde]
            for _ in range(2, K):
                LL.append(2 * L_tilde @ LL[-1] - LL[-2])
            return torch.stack(LL, dim=0)

def laplacianLambdaMax(L, normalization=None, is_undirected=False):
    eig_fn = eigs
    if is_undirected and normalization != 'rw':
        eig_fn = eigsh
    lambda_max = eig_fn(L, k=1, which='LM', return_eigenvectors=False)
    return float(lambda_max.real)


class ChebConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, num_supports: int = 1, bias: bool = False):
        super(ChebConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(num_supports, K, self.out_channels, self.in_channels)) 
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.permute(2, 3, 0, 1))
        if self.bias is not None:
            nn.init.uniform_(self.bias)
        
    def forward(self, x: torch.FloatTensor, L: torch.FloatTensor, **kwargs):
        r""" Chebyshev graph convolution operation
        Args:
            x (Tensor, [BNI]|[BTNI]): input time series data
            L (Tensor, [SNM]|[SBNM]|[NM]|[BNM]): adj matrix
            kwargs: any other specific parameters in dictionary
        Returns:
            Tensor, [BNO]|[BTNO]
        """
        if self.weight.shape[0] == 1 and L.shape[0] != 1: # S=1 and adj matrix of shape [NM]|[BNM]
            L = torch.unsqueeze(L, 0) #([NM]|[BNM])->[SNM]|[SBNM]
        else:
            assert self.weight.shape[0] == L.shape[0]
        if self.K > 1:
            batch_mul = (L.dim() == x.dim() == 4)
            h = None
            for L_, weight in zip(L, self.weight):
                x0 = x
                if h is None:
                    h = torch.tensordot(x0, weight[0], dims=([-1], [-1]))   # ([BNI]|[BTNI])[OI] -> ([BNO] | [BTNO])
                else:
                    h = h + torch.tensordot(x0, weight[0], dims=([-1], [-1]))
                if batch_mul:
                    x1 = torch.einsum("bnm,btni->btmi", L_, x)  #[BMN][BTNI]->[BTMI]
                else:
                    x1 = torch.matmul(L_.transpose(-1, -2), x)  # ([BNM]|[NM])->([BMN]|[MN]), [MN]([BNI]|[BTNI])->([BMI]|[BTMI]), [BMN][BNI] -> [BMI]
                h = h + torch.tensordot(x1, weight[1], dims=([-1], [-1])) # ([BNI]|[BTNI])[OI] -> ([BNO]|[BTNO])
                for k in range(2, self.K):
                    if batch_mul:
                        x2 = 2 * torch.einsum("bnm,btni->btmi", L_, x1) - x0
                    else:
                        x2 = 2 * torch.matmul(L_.transpose(-1, -2), x1) - x0
                    h = h + torch.tensordot(x2, weight[k], dims=([-1], [-1]))
                    x0, x1 = x1, x2
        else:
            weight = self.weight.squeeze(1)
            if L.dim() == 3:
                if x.dim() == 3:
                    rhs = torch.einsum("bni,snm->bism", x, L)
                elif x.dim() == 4:
                    rhs = torch.einsum("btni,snm->btism", x, L)
            elif L.dim() == 4:
                if x.dim() == 3:
                    rhs = torch.einsum("bni,sbnm->bism", x, L)
                elif x.dim() == 4:
                    rhs = torch.einsum("btni,sbnm->btism", x, L)
            if x.dim() == 3:
                h = torch.einsum("bism,soi->bmo", rhs, weight)
            elif x.dim() == 4:
                h = torch.einsum("btism,soi->btmo", rhs, weight)

        if self.bias is not None:
            h = h + self.bias # [BNO]|[BTNO]
            
        return h
        