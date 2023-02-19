import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from fuxits.predictor.predictor import Predictor
from fuxits.layers.graph import ChebConv

class gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {'GLU', 'relu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):
        """
        :param x: (3*N, B, Cin)
        :param mask:(3*N, 3*N)
        :return: (3*N, B, Cout)
        """
        adj = self.adj
        if mask is not None:
            adj = adj.to(mask.device) * mask

        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  # 3*N, B, Cin

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 3*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 3*N, B, Cout
            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs
            return out
        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 3*N, B, Cout


class STSGCM(nn.Module):
    def __init__(self, static_adj, in_dim, out_dims, num_nodes, activation='GLU'):
        """
        :param static_adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_nodes: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.static_adj = static_adj
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_nodes = num_nodes
        self.activation = activation
        self.gcn_operations = nn.ModuleList()
        self.gcn_operations.append(
            gcn_operation(
                adj=self.static_adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_nodes,
                activation=self.activation
            )
        )
        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.static_adj,
                    in_dim=self.out_dims[i-1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_nodes,
                    activation=self.activation
                )
            )

    def forward(self, x, mask=None):
        """
        :param x: (3N, B, Cin)
        :param mask: (3N, 3N)
        :return: (N, B, Cout)
        """
        need_concat = []
        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, mask)
            need_concat.append(x)
        # shape of each element is (1, N, B, Cout)
        need_concat = [
            torch.unsqueeze(
                h[self.num_nodes: 2 * self.num_nodes], dim=0
            ) for h in need_concat
        ]
        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)
        del need_concat
        return out


class STSGCL(nn.Module):
    def __init__(self, static_adj, hist_steps, num_nodes, in_dim, out_dims, strides=3, activation='GLU', temporal_emb=True, spatial_emb=True):
        """
        :param static_adj: 邻接矩阵
        :param hist_steps: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_nodes: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        self.static_adj = static_adj
        self.strides = strides
        self.hist_steps = hist_steps
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_nodes = num_nodes

        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        self.STSGCMS = nn.ModuleList()
        for i in range(self.hist_steps - self.strides + 1):
            self.STSGCMS.append(
                STSGCM(
                    static_adj=self.static_adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_nodes=self.num_nodes,
                    activation=self.activation
                )
            )
        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.hist_steps, 1, self.in_dim))
            # 1, T, 1, Cin
        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_nodes, self.in_dim))
            # 1, 1, N, Cin
        self.reset()

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)
        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
        """
        :param x: B, T, N, Cin
        :param mask: (N, N)
        :return: B, T-2, N, Cout
        """
        if self.temporal_emb:
            x = x + self.temporal_embedding
        if self.spatial_emb:
            x = x + self.spatial_embedding

        need_concat = []
        batch_size = x.shape[0]

        for i in range(self.hist_steps - self.strides + 1):
            t = x[:, i: i+self.strides, :, :]  # (B, 3, N, Cin)
            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_nodes, self.in_dim])
            # (B, 3*N, Cin)
            t = self.STSGCMS[i](t.permute(1, 0, 2), mask)  # (3*N, B, Cin) -> (N, B, Cout)
            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
            need_concat.append(t)

        out = torch.cat(need_concat, dim=1)  # (B, T-2, N, Cout)
        del need_concat, batch_size
        return out


class output_layer(nn.Module):
    def __init__(self, num_nodes, hist_steps, in_dim,
                 hidden_dim=128, pred_steps=12):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_nodes:节点数
        :param hist_steps:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param pred_steps:预测时间步长
        """
        super(output_layer, self).__init__()
        self.num_nodes = num_nodes
        self.hist_steps = hist_steps
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.pred_steps = pred_steps

        self.FC1 = nn.Linear(self.in_dim * self.hist_steps, self.hidden_dim, bias=True)
        self.FC2 = nn.Linear(self.hidden_dim, self.pred_steps, bias=True)

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)
        :return: (B, Tout, N)
        """
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin
        out1 = torch.relu(self.FC1(x.reshape(batch_size, self.num_nodes, -1)))
        # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden)
        out2 = self.FC2(out1)  # (B, N, hidden) -> (B, N, pred_steps)
        del out1, batch_size
        return out2.permute(0, 2, 1)  # B, pred_steps, N


class STSGCN(Predictor):
    def __init__(self, config, in_channels, num_nodes, hist_steps, pred_steps, static_adj, activation='GLU', use_mask=True, temporal_emb=True, spatial_emb=True, **kwargs):
        """
        :param in_channels:输入维度
        :param num_nodes:节点数量
        :param hist_steps:输入时间步长
        :param pred_steps:预测时间步长
        :param static_adj: local时空间矩阵
        :param hidden_dims: lists, 中间各STSGCL层的卷积操作维度
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param activation: 激活函数 {relu, GlU}
        :param use_mask: 是否使用mask矩阵对adj进行优化
        :param temporal_emb:是否使用时间嵌入向量
        :param spatial_emb:是否使用空间嵌入向量
        :param strides:滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        """
        super(STSGCN, self).__init__(config)
        self.num_nodes = num_nodes
        self.pred_steps = pred_steps
        self.hidden_dims = self.model_config['hidden_dims']
        self.first_layer_embedding_size = self.model_config['first_layer_embedding_size']
        self.out_layer_dim = self.model_config['out_layer_dim']
        self.strides = self.model_config['strides']
        self.activation = activation
        self.use_mask = use_mask
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.static_adj = self.construct_adj(static_adj, self.strides)

        self.First_FC = nn.Linear(in_channels, self.first_layer_embedding_size, bias=True)
        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                static_adj=self.static_adj,
                hist_steps=hist_steps,
                num_nodes=self.num_nodes,
                in_dim=self.first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )

        in_channels = self.hidden_dims[0][-1]
        hist_steps -= (self.strides - 1)

        for idx, hidden_list in enumerate(self.hidden_dims):
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(
                    static_adj=self.static_adj,
                    hist_steps=hist_steps,
                    num_nodes=self.num_nodes,
                    in_dim=in_channels,
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )

            hist_steps -= (self.strides - 1)
            in_channels = hidden_list[-1]

        self.predictLayer = nn.ModuleList()
        for t in range(self.pred_steps):
            self.predictLayer.append(
                output_layer(
                    num_nodes=self.num_nodes,
                    hist_steps=hist_steps,
                    in_dim=in_channels,
                    hidden_dim=self.out_layer_dim,
                    pred_steps=1
                )
            )
        if self.use_mask:
            mask = torch.zeros_like(self.static_adj)
            mask[self.static_adj != 0] = self.static_adj[self.static_adj != 0]
            self.mask = nn.Parameter(mask)
        else:
            self.mask = None


    def construct_adj(self, A, steps):
        """
        构建local 时空图
        :param A: adjacency matrix, shape is (N, N)
        :param steps: 选择几个时间步来构建图
        :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
        """
        N = len(A)  # 获得行数
        adj = np.zeros((N * steps, N * steps))

        for i in range(steps):
            """对角线代表各个时间步自己的空间图，也就是A"""
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

        for i in range(N):
            for k in range(steps - 1):
                """每个节点只会连接相邻时间步的自己"""
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1

        for i in range(len(adj)):
            """加入自回"""
            adj[i, i] = 1
        return torch.FloatTensor(adj)

    def forward(self, x, y=None, batch_idx=None):
        """
        :param x: B, Tin, N, Cin)
        :return: B, Tout, N, 1
        """
        x = torch.relu(self.First_FC(x))  # B, Tin, N, Cin
        for model in self.STSGCLS:
            x = model(x, self.mask)
        # (B, T - 8, N, Cout)
        need_concat = []
        for i in range(self.pred_steps):
            out_step = self.predictLayer[i](x)  # (B, 1, N)
            need_concat.append(out_step)
        out = torch.cat(need_concat, dim=1)  # B, Tout, N
        out = out.unsqueeze(-1)
        del need_concat
        return out
