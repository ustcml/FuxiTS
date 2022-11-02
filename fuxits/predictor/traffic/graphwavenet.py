import math
from fuxits.predictor.predictor import Predictor
import torch
import torch.nn as nn
from fuxits.layers.graph import ChebConv
from fuxits.layers.graph import compute_graph_poly
import torch.nn.functional as F
class GraphWaveNet(Predictor):
    
    def __init__(self, config, in_channels, num_nodes, hist_steps, pred_steps, static_adj, **kwargs):
        super(GraphWaveNet, self).__init__(config)
        self.gcn_order = self.model_config.pop('gcn_order')
        ada_graph_emb_dim = self.model_config.pop('ada_graph_emb_dim')
        assert self.gcn_order is not None or ada_graph_emb_dim is None
        mlp_hidden_dim = self.model_config.pop('mlp_hidden_dim')
        tcn_input_dim = self.model_config['tcn_input_dim']
        block_output_dim = self.model_config['block_output_dim']
        num_layers = self.model_config['num_layers']
        tcn_kernel = self.model_config['tcn_kernel']
        self.receptive_field = (tcn_kernel - 1) * (2**num_layers - 1)
        self.num_blocks = math.ceil((hist_steps - in_channels) / self.receptive_field)
        self.receptive_field = self.receptive_field * self.num_blocks + in_channels
        cheb = compute_graph_poly(static_adj, self.gcn_order+1, 'rw', True)
        chebt = compute_graph_poly(static_adj.T, self.gcn_order+1, 'rw', True, False)
        self.register_buffer('supports', torch.cat([cheb, chebt]))
        if ada_graph_emb_dim is not None:
            self.ada_graph_emb_1 = nn.Parameter(torch.randn(num_nodes, ada_graph_emb_dim))
            self.ada_graph_emb_2 = nn.Parameter(torch.randn(num_nodes, ada_graph_emb_dim))
            num_supports = len(self.supports) + self.gcn_order
        else:
            self.register_parameter('ada_graph_emb_1', None)
            self.register_parameter('ada_graph_emb_2', None)
            num_supports = len(self.supports)
        
        self.linear = nn.Linear(in_channels, tcn_input_dim)
        self.blocks = nn.ModuleList(
            GWaveBlock(num_supports=num_supports, gcn_order=1, **self.model_config) for _ in range(self.num_blocks))
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(block_output_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, pred_steps)
        )
    

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, batch_idx=None):
        if self.ada_graph_emb_1 is not None:
            ada_graph = F.softmax(F.relu(self.ada_graph_emb_1 @ self.ada_graph_emb_2.T), dim=1)
            ada_graph = compute_graph_poly(ada_graph, self.gcn_order+1, ret_idt_mat=False)
            supports = torch.cat([self.supports, ada_graph])
        else:
            supports = self.supports
        if x.size(1) < self.receptive_field:
            x = F.pad(x,(0, 0, 0, 0, self.receptive_field - x.size(1), 0))
        inputs = self.linear(x)
        outputs = None
        for block in self.blocks:
            inputs, block_output = block(inputs, supports)
            if outputs is None:
                outputs = block_output
            else:
                outputs = outputs[:,-block_output.size(1):,:,:] + block_output
        return self.mlp(outputs).transpose(1, 3)



class GWaveBlock_OneLayer(nn.Module):
    def __init__(self, 
        tcn_input_dim,
        tcn_output_dim,
        tcn_kernel,
        block_output_dim,
        tcn_dilation,
        num_supports,
        order=None,
        drop_ratio=0.0
    ) -> None:
        super().__init__()
        self.tcn = nn.Conv2d(in_channels=tcn_input_dim, out_channels=tcn_output_dim*2, kernel_size=(1, tcn_kernel), dilation=tcn_dilation)
        if order is not None:
            self.hidden = ChebConv(tcn_output_dim, tcn_input_dim, order, num_supports, True)
        else:
            self.hidden = nn.Linear(tcn_output_dim, tcn_input_dim)
        self.dropout = nn.Dropout(drop_ratio)
        self.bn = nn.BatchNorm2d(tcn_input_dim)
        self.linear = nn.Linear(tcn_output_dim, block_output_dim)
    
    def forward(self, x: torch.Tensor, ada_graph=None):
        """
        Args:
            x (Tensor, [BTNI]): input data
        Returns:
            residual (Tensor, [BTNI])
            skip (Tensor, [BTNI])
        """
        output = self.tcn(x.transpose(1, 3)).transpose(1, 3)
        value, gate = torch.chunk(output, 2, -1)
        tcn_output = torch.tanh(value) * torch.sigmoid(gate)
        if isinstance(self.hidden, ChebConv):
            gcn_output = self.hidden(tcn_output, ada_graph)
        else:
            gcn_output = self.hidden(tcn_output)
        output = self.dropout(gcn_output) + x[:,-gcn_output.size(1):,:,:]
        return self.bn(output.transpose(1,3)).transpose(1,3), self.linear(tcn_output)


class GWaveBlock(nn.Module):

    def __init__(self, num_layers,
        tcn_input_dim,
        tcn_output_dim,
        tcn_kernel,
        block_output_dim,
        num_supports,
        gcn_order,
        drop_ratio
    ) -> None:
        super(GWaveBlock, self).__init__()
        self.layers = nn.ModuleList(
            GWaveBlock_OneLayer(
            tcn_input_dim,
            tcn_output_dim,
            tcn_kernel,
            block_output_dim,
            2**i,
            num_supports, 
            gcn_order,
            drop_ratio) for i in range(num_layers)
        )
    
    def forward(self, x: torch.Tensor, ada_graph=None):
        inputs = x
        block_output = None
        for gwave in self.layers:
            inputs, tcn_output = gwave(inputs, ada_graph)
            if block_output is None:
                block_output = tcn_output
            else:
                block_output = block_output[:,-tcn_output.size(1):,:,:] + tcn_output
        return inputs, block_output
