from fuxits.layers.graph import ChebConv as GCC
from torch_geometric.nn import ChebConv
from torch_sparse import SparseTensor
from torch_geometric.utils import dense_to_sparse
import torch, numpy as np
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul, remove_diag, set_diag, get_diag, cat
from fuxits.layers.graph import get_laplacian, get_normalization, remove_selfloops, scale_laplacian, to_scipy_sparse_matrix, laplacianLambdaMax
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from pytorch_lightning import seed_everything
num_nodes = 5
adj = torch.randn(num_nodes, num_nodes)
adj[adj<0] = 0
adj = adj + adj.T
adj.fill_diagonal_(0)
edge_idx = adj.nonzero().T
edge_weight = adj[edge_idx[0], edge_idx[1]]
sadj = SparseTensor.from_dense(adj)

edge_idx1, edge_weight1 = adj.t(), None
edge_idx2, edge_weight2 = sadj.t(), None
normalization = 'sym'
a1 = to_scipy_sparse_matrix(*get_laplacian(edge_idx, edge_weight, normalization, num_nodes=num_nodes), num_nodes)
a2 = to_scipy_sparse_matrix(*get_laplacian(edge_idx1, edge_weight1, normalization, num_nodes=num_nodes), num_nodes)
a3 = to_scipy_sparse_matrix(*get_laplacian(edge_idx2, edge_weight2, normalization, num_nodes=num_nodes), num_nodes)
print(np.abs(a1-a2.T).sum())
print(np.abs(a1-a3.T).sum())
print(np.abs(a2-a3).sum())
normalization = None
a1 = to_scipy_sparse_matrix(*get_laplacian(edge_idx, edge_weight, normalization, num_nodes=num_nodes), num_nodes)
a2 = to_scipy_sparse_matrix(*get_laplacian(edge_idx1, edge_weight1, normalization, num_nodes=num_nodes), num_nodes)
a3 = to_scipy_sparse_matrix(*get_laplacian(edge_idx2, edge_weight2, normalization, num_nodes=num_nodes), num_nodes)
print(np.abs(a1-a2.T).sum())
print(np.abs(a1-a3.T).sum())
print(np.abs(a2-a3).sum())
normalization = 'rw'
a1 = to_scipy_sparse_matrix(*get_laplacian(edge_idx, edge_weight, normalization, num_nodes=num_nodes), num_nodes)
a2 = to_scipy_sparse_matrix(*get_laplacian(edge_idx1, edge_weight1, normalization, num_nodes=num_nodes), num_nodes)
a3 = to_scipy_sparse_matrix(*get_laplacian(edge_idx2, edge_weight2, normalization, num_nodes=num_nodes), num_nodes)
print(np.abs(a1-a2.T).sum())
print(np.abs(a1-a3.T).sum())
print(np.abs(a2-a3).sum())

normalization = 'sym'
a1 = laplacianLambdaMax(*get_laplacian(edge_idx, edge_weight, normalization, num_nodes=num_nodes), num_nodes, normalization)
a2 = laplacianLambdaMax(*get_laplacian(edge_idx1, edge_weight1, normalization, num_nodes=num_nodes), num_nodes, normalization)
a3 = laplacianLambdaMax(*get_laplacian(edge_idx2, edge_weight2, normalization, num_nodes=num_nodes), num_nodes, normalization)
print(abs(a1- a2), abs(a1-a3))
normalization = None
a1 = laplacianLambdaMax(*get_laplacian(edge_idx, edge_weight, normalization, num_nodes=num_nodes), num_nodes, normalization)
a2 = laplacianLambdaMax(*get_laplacian(edge_idx1, edge_weight1, normalization, num_nodes=num_nodes), num_nodes, normalization)
a3 = laplacianLambdaMax(*get_laplacian(edge_idx2, edge_weight2, normalization, num_nodes=num_nodes), num_nodes, normalization)
print(abs(a1- a2), abs(a1-a3))
normalization = 'rw'
a1 = laplacianLambdaMax(*get_laplacian(edge_idx, edge_weight, normalization, num_nodes=num_nodes), num_nodes, normalization)
a2 = laplacianLambdaMax(*get_laplacian(edge_idx1, edge_weight1, normalization, num_nodes=num_nodes), num_nodes, normalization)
a3 = laplacianLambdaMax(*get_laplacian(edge_idx2, edge_weight2, normalization, num_nodes=num_nodes), num_nodes, normalization)
print(abs(a1- a2), abs(a1-a3))

a1 = to_scipy_sparse_matrix(*get_normalization(edge_idx, edge_weight, num_nodes, fill_value=1., symmetry=True), num_nodes)
a2 = to_scipy_sparse_matrix(*get_normalization(edge_idx1, edge_weight1, num_nodes, fill_value=1., symmetry=True), num_nodes)
a3 = to_scipy_sparse_matrix(*get_normalization(edge_idx2, edge_weight2, num_nodes, fill_value=1., symmetry=True), num_nodes)
print(np.abs(a1-a2.T).sum())
print(np.abs(a1-a3.T).sum())
print(np.abs(a2-a3).sum())

a1 = to_scipy_sparse_matrix(*get_normalization(edge_idx, edge_weight, num_nodes, fill_value=2., symmetry=True), num_nodes)
a2 = to_scipy_sparse_matrix(*get_normalization(edge_idx1, edge_weight1, num_nodes, fill_value=2., symmetry=True), num_nodes)
a3 = to_scipy_sparse_matrix(*get_normalization(edge_idx2, edge_weight2, num_nodes, fill_value=2., symmetry=True), num_nodes)
print(np.abs(a1-a2.T).sum())
print(np.abs(a1-a3.T).sum())
print(np.abs(a2-a3).sum())


a1 = to_scipy_sparse_matrix(*get_normalization(edge_idx, edge_weight, num_nodes, fill_value=1., symmetry=False), num_nodes)
a2 = to_scipy_sparse_matrix(*get_normalization(edge_idx1, edge_weight1, num_nodes, fill_value=1., symmetry=False), num_nodes)
a3 = to_scipy_sparse_matrix(*get_normalization(edge_idx2, edge_weight2, num_nodes, fill_value=1., symmetry=False), num_nodes)
print(np.abs(a1-a2.T).sum())
print(np.abs(a1-a3.T).sum())
print(np.abs(a2-a3).sum())

seed_everything(20, workers=True)
num_nodes = 5
adj = torch.randn(num_nodes, num_nodes)
adj[adj<0] = 0
adj = adj + adj.T
adj.fill_diagonal_(0)
edge_idx = adj.nonzero().T
edge_weight = adj[edge_idx[0], edge_idx[1]]
sadj = SparseTensor.from_dense(adj)
x = torch.randn(30, 20, 5, 10) * 10

normalization=None
seed_everything(20, workers=True)
lm = laplacianLambdaMax(*get_laplacian(edge_idx, edge_weight, normalization, num_nodes=num_nodes), num_nodes, normalization)
conv = GCC(10, 2, 3, normalization=normalization)
y = conv(x, edge_idx, edge_weight)
y1 = conv(x, SparseTensor.from_dense(adj).t())
seed_everything(20, workers=True)
conv = ChebConv(10, 2, 3, normalization=normalization)
y2 = conv(x, edge_idx, edge_weight, lambda_max = lm)
print(torch.abs(y- y2).mean().detach().numpy())
print(torch.abs(y- y1).mean().detach().numpy())

normalization='rw'
seed_everything(20, workers=True)
lm = laplacianLambdaMax(*get_laplacian(edge_idx, edge_weight, normalization, num_nodes=num_nodes), num_nodes, normalization)
conv = GCC(10, 2, 3, normalization=normalization)
y = conv(x, edge_idx, edge_weight)
y1 = conv(x, SparseTensor.from_dense(adj).t())
seed_everything(20, workers=True)
conv = ChebConv(10, 2, 3, normalization=normalization)
y2 = conv(x, edge_idx, edge_weight, lambda_max = lm)
print(torch.abs(y- y2).mean().detach().numpy())
print(torch.abs(y- y1).mean().detach().numpy())

normalization='sym'
seed_everything(20, workers=True)
lm = laplacianLambdaMax(*get_laplacian(edge_idx, edge_weight, normalization, num_nodes=num_nodes), num_nodes, normalization)
conv = GCC(10, 2, 3, normalization=normalization)
y = conv(x, edge_idx, edge_weight)
y1 = conv(x, SparseTensor.from_dense(adj).t())
seed_everything(20, workers=True)
conv = ChebConv(10, 2, 3, normalization=normalization)
y2 = conv(x, edge_idx, edge_weight, lambda_max = lm)
print(torch.abs(y- y2).mean().detach().numpy())
print(torch.abs(y- y1).mean().detach().numpy())

# adj = torch.randn(3, 3)
# adj[adj<0] = 0
# sadj = SparseTensor.from_dense(adj)
# zero = SparseTensor(row=torch.zeros(0, dtype=torch.long), col=torch.zeros(0, dtype=torch.long), value=torch.zeros(0), sparse_sizes=(3,3))
# x = cat([zero, sadj], dim=0)
# xx = cat([sadj.t(), zero], dim=0)
# xxx = cat([x, xx], dim=1)
# print(xxx.to_dense())