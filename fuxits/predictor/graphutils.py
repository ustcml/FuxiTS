import torch
from scipy.sparse.linalg import eigs

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
        L = degree(adj_t) - adj_t
    return L

def scale_lapacian(L:torch.Tensor):
    lambda_max = eigs(L.cpu().numpy(), k=1, which='LR')[0].real
    return (2 * L) / lambda_max - torch.eye(L.shape[0])
