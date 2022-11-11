from torch_geometric.nn import MessagePassing, Linear
from typing import Optional, Union
from torch_geometric.typing import OptTensor
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian as get_laplacian_index
from torch_geometric.utils import remove_self_loops, add_self_loops, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul, remove_diag, set_diag, get_diag
from scipy.sparse.linalg import eigs, eigsh
import numpy as np, scipy

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

    adj = symmetrize(adj, kwargs.pop('symmetrize_aggregation', None))
    process_funs = {'binary':binary_graph, 'epsilon':epsilon_graph}
    output = process_funs[mode](adj, **kwargs)
    return torch.from_numpy(output).type(torch.float32)

def compute_graph_poly(adj, K, norm=None, lap=True, scaled=True, add_self_loop=False, ret_idt_mat=True):
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj).type(torch.float32)
    if lap:
        if add_self_loop:
            adj = adj.fill_diagonal_(1.0)
        L_tilde, _ = get_laplacian(adj.T, None, normalization=norm)
        if scaled:
            lambda_max = laplacianLambdaMax(L_tilde.T, normalization=norm)
            L_tilde, _ = scale_laplacian(L_tilde, lambda_max=lambda_max)
        L_tilde = L_tilde.T
    else:
        if norm is not None:
            L_tilde, _ = get_normalization(adj.T, fill_value=1.0 if add_self_loop else None, symmetry=(norm=='sym'))
            L_tilde = L_tilde.T
        else:
            L_tilde = adj
    
    if K == 1:
        return L_tilde
    else:
        LL = [torch.eye(L_tilde.shape[0]).type_as(L_tilde), L_tilde]
        for _ in range(2, K):
            LL.append(L_tilde @ LL[-1])
        if ret_idt_mat:
            return torch.stack(LL, dim=0)
        else:
            return torch.stack(LL[1:], dim=0)

def compute_cheb_poly(adj, K, norm=None, lap=True, scaled=True, add_self_loop=False, ret_idt_mat=True):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Args:
        adj (Tensor or ndarray, [NN]): input adjacency matrix
        norm (str, 'sym'|'rw'|None): how to normalize adjacency matrix
        lap (bool): whether computing laplacian matrix
        scaled (bool): whether scale output matrix
    Returns
        cheb_polynomials (Tensor, [KNN] if ret_idt_mat else [K-1 NN])
    '''
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj).type(torch.float32)
    if lap:
        if add_self_loop:
            adj = adj.fill_diagonal_(1.0)
        L_tilde, _ = get_laplacian(adj.T, None, normalization=norm)
        if scaled:
            lambda_max = laplacianLambdaMax(L_tilde.T, normalization=norm)
            L_tilde, _ = scale_laplacian(L_tilde, lambda_max=lambda_max)
        L_tilde = L_tilde.T
    else:
        if norm is not None:
            L_tilde, _ = get_normalization(adj.T, fill_value=1.0 if add_self_loop else None, symmetry=(norm=='sym'))
            L_tilde = L_tilde.T
        else:
            L_tilde = adj

    if K == 1:
        return L_tilde
    else:
        LL = [torch.eye(L_tilde.shape[0]).type_as(L_tilde), L_tilde]
        for _ in range(2, K):
            LL.append(2 * L_tilde @ LL[-1] - LL[-2])
        if ret_idt_mat:
            return torch.stack(LL, dim=0)
        else:
            return torch.stack(LL[1:], dim=0)


def add_selfloops(edge_index, edge_weight=None, fill_value=1., num_nodes: Optional[int] = None):
    if isinstance(edge_index, SparseTensor):
        return fill_diag(edge_index, fill_value), None
    elif edge_index.size(0) == edge_index.size(1):
        return edge_index.fill_diagonal_(fill_value), None
    else:
        return add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

def remove_selfloops(edge_index, edge_weight=None):
    if isinstance(edge_index, SparseTensor):
        return remove_diag(edge_index), None
    elif edge_index.size(0) == edge_index.size(1):
        return edge_index.fill_diagonal_(0.), None
    else:
        return remove_self_loops(edge_index, edge_weight)



def to_scipy_sparse_matrix(edge_index, edge_weight=None, num_nodes=None):
    r"""Converts a graph given by torch_sparse.SparseTensor to a scipy sparse matrix.
    Args:
        A (SparseTensor): a graph adjacency matrix
    Returns:
        The corresponding scipy sparse matrix
    """
    if isinstance(edge_index, SparseTensor):
        row, col, val = edge_index.coo()
        num_nodes = edge_index.size(0)
    elif edge_index.size(0) == edge_index.size(1):
        row, col = edge_index.nonzero(as_tuple=True)
        val = edge_index[row, col]
        num_nodes = edge_index.size(0)
    else:
        row, col = edge_index
        val = edge_weight
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if val is None:
        val = torch.ones(row.size(0))
    N = num_nodes
    out = scipy.sparse.coo_matrix(
        (val.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())), (N, N))
    return out

def laplacianLambdaMax(edge_index, edge_weight=None, num_nodes=None, normalization=None):
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    eig_fn = eigs
    if normalization != 'rw':
        eig_fn = eigsh
    lambda_max = eig_fn(L, k=1, which='LM', return_eigenvectors=False)
    return float(lambda_max.real)


def negative(val: SparseTensor):
    return val.set_value(val.storage.value() * -1, layout='coo')

def sparsemul(src: SparseTensor, other: Union[torch.Tensor, float, int]) -> SparseTensor:
    if isinstance(other, Tensor) and other.dim()>0:
        return mul(src, other)
    else:
        return src.set_value(src.storage.value() * other, layout='coo')
        
    
SparseTensor.__neg__ = negative
SparseTensor.__mul__ = sparsemul
SparseTensor.__rmul__ = sparsemul


def get_normalization(edge_index, edge_weight=None, num_nodes=None, fill_value: Optional[float]=1., symmetry=True, dtype=None):
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if fill_value is not None:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=0)
        if symmetry:
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
            adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
            return adj_t, None
        else:
            deg_inv = 1/deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
            adj_t = mul(adj_t, deg_inv.view(1, -1))
            return adj_t, None
    elif edge_index.size(0) == edge_index.size(1):
        adj_t = edge_index
        if fill_value is not None:
            adj_t = adj_t.fill_diagonal_(fill_value)
        deg = torch.sum(adj_t, dim=0)
        if symmetry:
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
            adj_t = adj_t * deg_inv_sqrt.view(1, -1)
            return adj_t, None
        else:
            deg_inv = 1/deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
            adj_t = adj_t * deg_inv.view(1, -1)
            return adj_t, None
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        if fill_value is not None:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        if symmetry:
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            deg_inv = 1/deg
            deg_inv.masked_fill_(deg_inv == float('inf'), 0)
            return edge_index, deg_inv[row] * edge_weight

def scale_laplacian(edge_index, edge_weight=None, lambda_max=None, num_nodes: Optional[int] = None):
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        adj_t = (2.0/lambda_max) * adj_t
        adj_t = set_diag(adj_t, get_diag(adj_t)-1)
        return adj_t, None
    elif edge_index.size(0) == edge_index.size(1):
        adj_t = edge_index
        adj_t = (2.0/lambda_max) * adj_t
        adj_t = adj_t - torch.eye(adj_t.size(0))
        return adj_t, None
    else:
        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                fill_value=-1.,
                                                num_nodes=num_nodes)
        assert edge_weight is not None
        return edge_index, edge_weight

def get_laplacian(edge_index, edge_weight=None, normalization=None, dtype: Optional[int] = None, num_nodes: Optional[int] = None):
    if isinstance(edge_index, SparseTensor):
        return get_laplacian_sparse(edge_index, normalization, dtype), None
    elif edge_index.size(0) == edge_index.size(1):
        return get_laplacian_dense(edge_index, normalization, dtype), None
    else:
        return get_laplacian_index(edge_index, edge_weight, normalization, dtype, num_nodes)

def get_laplacian_dense(adj_t:Tensor, normalization=None, dtype: Optional[int] = None):
    deg = torch.sum(adj_t, dim=0)
    if normalization is None:
        # L = D - A.
        adj_t = torch.diag(deg) - adj_t
    elif normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
        adj_t = adj_t * deg_inv_sqrt.view(1, -1)
        # L = I - A_norm.
        adj_t = torch.eye(len(deg)) - adj_t
    else:
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        adj_t = adj_t * deg_inv.view(1, -1)
        # L = I - A_norm.
        adj_t = torch.eye(len(deg)) - adj_t
    return adj_t

def get_laplacian_sparse(adj_t, normalization=None, dtype: Optional[int] = None):
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1., dtype=dtype)
    deg = sparsesum(adj_t, dim=0)
    if normalization is None:
        # L = D - A.
        adj_t = set_diag(-adj_t, deg)
    elif normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        # L = I - A_norm.
        adj_t = fill_diag(-adj_t, 1.0)
    else:
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        adj_t = mul(adj_t, deg_inv.view(1, -1))
        # L = I - A_norm.
        adj_t = fill_diag(-adj_t, 1.0)
    return adj_t

class ChebConv(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
        \mathbf{\Theta}^{(k)}

    where :math:`\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}

        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int, K: int,
                 normalization: Optional[str] = 'sym', bias: bool = True, cached: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.cached = cached
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
    
    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        edge_index, edge_weight = remove_selfloops(edge_index, edge_weight)
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization, dtype, num_nodes)
        if lambda_max is None:
            lambda_max = laplacianLambdaMax(edge_index, edge_weight, num_nodes, normalization)
        return scale_laplacian(edge_index, edge_weight, lambda_max, num_nodes)

        
       

    def forward(self, x, edge_index, edge_weight: OptTensor = None,
                batch: OptTensor = None, lambda_max: OptTensor = None):
        """"""
        if self._cached_edge_index is None:
            if isinstance(edge_index, SparseTensor):
                edge_index = edge_index.to_device(x.device)
            else:
                edge_index.to(x.device)
            if edge_weight is not None:
                edge_weight.to(x.device)
            edge_index, edge_weight = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype,
                                         batch=batch)
            if self.cached:
                self._cached_edge_index = (edge_index, edge_weight)
        else:
            edge_index, edge_weight = self._cached_edge_index


        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, edge_weight=edge_weight, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            len(self.lins), self.normalization)




class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_supports: int = 1, bias: bool = False):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(num_supports, self.out_channels, self.in_channels)) 
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.permute(1, 2, 0))
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

        weight = self.weight
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
        