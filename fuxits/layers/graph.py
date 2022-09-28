import torch
from scipy.sparse.linalg import eigs
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def preprocess_graph(mode:str, adj, **kwargs):
    def binary_graph(adj, epsilon=np.finfo(float).eps):
        adj[adj > epsilon] = 1
        return adj

    def epsilon_graph(adj, sigma=0.1, epsilon=0.5):
        idx = adj < np.finfo(float).eps
        output = np.exp(-adj * adj / sigma) 
        output[(output < epsilon) | idx] = 0.
        np.fill_diagonal(output, 0.)
        return output

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
    return torch.fill_diag(adj_t, weight)

def laplacian(adj_t, normalized=None, add_self_loops=False):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: torch.tensor, shape is (N, N), N is the num of vertices
    Returns
    normalized: 'sys' and 'rw'
    ----------
    laplacian: torch.tensor, shape (N, N)
    '''

    if normalized:
        assert normalized in {'sys', 'rw'}
        W = graph_adj_norm(W, normalized=='sys', add_self_loops)
        return torch.eye(W.shape[0]) - W
    else:
        if add_self_loops:
            adj_t = add_self_loops_fun(adj_t)
        L = torch.diag(degree(adj_t)) - adj_t
    return L

def scale_lapacian(L:torch.Tensor):
    lambda_max = eigs(L.cpu().numpy(), k=1, which='LR')[0].real
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
        L_tilde = scale_lapacian(laplacian(adj, normalized=normalized, add_self_loops=add_self_loop))
        LL = [torch.eye(L_tilde.shape[0]), L_tilde]
        for _ in range(2, K):
            LL.append(2 * L_tilde * LL[-1] - LL[-2])
        return torch.stack(LL, dim=-1)


class ChebConv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, in_channels, out_channels, K, static_adj, bias=False):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(ChebConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.lins = torch.nn.ModuleList([
        #    nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)
        #])
        self.lins = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_channels, 1, 1))

        if isinstance(static_adj, torch.Tensor) and static_adj.dim() == 3:
            self.register_buffer('chebpoly', static_adj)
            #self.chebpoly = static_adj
        else:
            #self.chebpoly = torch.stack(list(compute_cheb_poly(static_adj, K)))
            self.register_buffer('chebpoly', compute_cheb_poly(static_adj, K))

    def forward(self, x, dyna_adj=None):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, F_in, N, T)
        :return: (batch_size, F_out, N, T)
        ''' 
        if dyna_adj is not None:
            #B, N = x.shape[:2]
            #rhs = torch.bmm(Tk.reshape(B, N, -1).transpose(1, 2), x.reshape(B, N, -1)).view(B, N, -1, x.shape[-1]) # B N FT x  B N MK -> B MK FT -> B M KF T 
            #output = torch.matmul(rhs.transpose(-1, -2), self.lins).transpose(-1, -2)
            Tk = torch.unsqueeze(self.chebpoly, 0) * torch.unsqueeze(dyna_adj, -1) # B N M K
            rhs = torch.einsum('bfnt,bnmk->bftmk', x, Tk)
        else:
            rhs = torch.einsum('bfnt,nmk->bftmk', x, self.chebpoly)
        
        output = torch.einsum('bftmk,kfo->bomt', rhs, self.lins)

        if hasattr(self, 'bias'):
            output = output + self.bias

        return output
