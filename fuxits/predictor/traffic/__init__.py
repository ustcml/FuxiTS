from fuxits.predictor.traffic.astgcn import ASTGCN
from fuxits.predictor.traffic.stgcn import STGCN
#import torch
#import torch.nn as nn
#from torch_geometric.nn import MessagePassing

#class GNNAdapter(nn):
#    def __init__(self, gnn: MessagePassing) -> None:
#        super().__init__()
#        self.gnn = gnn

    
    # def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs):
    #     '''
    #     :param x: (batch_size, N, F_in, T)
    #     :return: (batch_size, N, F_out, T)
    #     '''
    #     N = x.shape[0]
    #     x = x.transpose(0, 1)
    #     x_ = self.gnn(x.view(N, -1), edge_index, kwargs)
    #     return x_.view(*x.shape).transpose(0, 1)

