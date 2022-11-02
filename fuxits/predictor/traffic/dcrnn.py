import math, random
from fuxits.predictor.predictor import Predictor
from fuxits.layers.graph import ChebConv
import torch.nn as nn
import torch
from fuxits.layers.graph import compute_cheb_poly


class DCRNN(Predictor):
    
    def __init__(self, config, in_channels, num_nodes, hist_steps, pred_steps, static_adj, **kwargs):
        super(DCRNN, self).__init__(config)
        cheb = compute_cheb_poly(static_adj, self.model_config['max_diffusion_step']+1, 'rw', True)
        chebt = compute_cheb_poly(static_adj.T, self.model_config['max_diffusion_step']+1, 'rw', True, False)
        dcgru_args = {
            "supports": torch.cat([cheb, chebt]),
            "num_nodes": num_nodes,
            "num_layers": self.model_config['num_layers'],
            "hidden_dim": self.model_config['hidden_dim'],
            "max_diffusion_step": 0,
            "use_gc_for_ru": self.model_config['use_gc_for_ru'],
        }
        self.encoder = DCGRUEncoder(in_channels, hist_steps, **dcgru_args)
        self.decoder = DCGRUDecoder(in_channels, pred_steps, 
                                    self.model_config['use_scheduled_sampling'], 
                                    self.model_config['cl_decay_steps'], 
                                    **dcgru_args)

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, batch_idx=None):
        encoder_hidden_state = self.encoder(x.transpose(0, 1))
        if y is not None:
            outputs = self.decoder(encoder_hidden_state, y.transpose(0, 1), self.global_step)
        else:
            outputs = self.decoder(encoder_hidden_state, y, self.global_step)
        return outputs.transpose(0, 1)



class DCGRUEncoder(nn.Module):
    def __init__(self, input_dim, seq_len, **kwargs):
        super(DCGRUEncoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = kwargs['hidden_dim']
        self.dcgru = DCGRUCell(input_dim=input_dim, **kwargs)

    def forward(self, inputs:torch.Tensor):
        """
        Args
            inputs: (Tensor, [BTNI])
        Return
            hidden state: (Tensor, [LBNF])
        """
        encoder_hidden_state = torch.zeros(len(self.dcgru.layers), *inputs.shape[1:-1], self.hidden_dim).type_as(inputs) # [LBNI]
        for input in inputs:
            _, encoder_hidden_state = self.dcgru(input, encoder_hidden_state)
        return encoder_hidden_state


class DCGRUDecoder(nn.Module):
    def __init__(self, in_channels, pred_steps, use_scheduled_sampling, cl_decay_steps, **kwargs):
        super().__init__()
        self.pred_steps = pred_steps
        self.use_scheduled_sampling = use_scheduled_sampling
        self.cl_decay_steps = cl_decay_steps
        self.output_dim = in_channels
        self.dcgru = DCGRUCell(input_dim=in_channels, **kwargs)
        self.projection_layer = nn.Linear(kwargs["hidden_dim"], in_channels)
    
    def _compute_sampling_threshold(self, batches_seen):
       return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))

    def forward(self, encoder_state, y, batch_idx):
        """ Decoder based on DCGRUCell
        Args:
            encoder_state (Tensor, [LBNF]): hidden_state of encoders
            y (Tensor, [BTNI]): target time series
            batch_idx (integer): the number of batched processed
        Returns:
            Tensor, the same shape as y
        """
        decoder_state = encoder_state #[LBNF]
        decoder_input = torch.zeros(*encoder_state.shape[1:-1], self.output_dim).type_as(decoder_state) # [BNI]
        outputs = []
        for t in range(self.pred_steps):
            decoder_output, decoder_state = self.dcgru(decoder_input, decoder_state)
            decoder_output = self.projection_layer(decoder_output)
            outputs.append(decoder_output)
            decoder_input = decoder_output
            if self.training and self.use_scheduled_sampling:
                if random.random() < self._compute_sampling_threshold(batch_idx):
                    decoder_input = y[t]
        outputs = torch.stack(outputs)
        return outputs


class DCGRUCell_OneLayer(nn.Module):
    def __init__(self,
        num_nodes,
        input_dim,
        hidden_dim,
        supports,
        max_diffusion_step,
        activation=torch.tanh,
        use_gc_for_ru=True
    ):
        super(DCGRUCell_OneLayer, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._hidden_dim = hidden_dim
        self._use_gc_for_ru = use_gc_for_ru
        self.register_buffer("_supports", supports)
        K = max_diffusion_step + 1
        num_supports = len(supports)
        if use_gc_for_ru:
            self.reset_update = ChebConv(input_dim+hidden_dim, 2*hidden_dim, K, num_supports, bias=True)
        else:
            self.reset_update = nn.Linear(input_dim+hidden_dim, 2*hidden_dim)
        self.hidden = ChebConv(input_dim+hidden_dim, hidden_dim, K, num_supports, bias=True)
        

    def forward(self, inputs, state):
        x = torch.cat([inputs, state], dim=-1)
        if self._use_gc_for_ru:
            gate = torch.sigmoid(self.reset_update(x, self._supports)) # [BNI][SKNM]->[BISKM][SKOI]->[BMO]
        else:
            gate = torch.sigmoid(self.reset_update(x)) # [BNI][OI]->[BNO]
        reset_gate, update_gate = gate.chunk(2, -1)
        x = torch.cat([inputs, reset_gate * state], dim=-1)
        new_state = self.hidden(x, self._supports)
        if self._activation is not None:
            new_state = self._activation(new_state)
        return update_gate * state +  (1.0 - update_gate) * new_state

class DCGRUCell(nn.Module):
    def __init__(self,
        num_layers,
        num_nodes,
        input_dim,
        hidden_dim,
        supports,
        max_diffusion_step,
        activation=torch.tanh,
        use_gc_for_ru=True,
    ):
        super(DCGRUCell, self).__init__()
        self.layers = nn.ModuleList(
            DCGRUCell_OneLayer(num_nodes, 
            input_dim if i==0 else hidden_dim, 
            hidden_dim, 
            supports, 
            max_diffusion_step, 
            activation, 
            use_gc_for_ru) for i in range(num_layers)
        )

    def forward(self, inputs, layers_state):
        hidden_states = []
        output = inputs
        for dcgrucell, state in zip(self.layers, layers_state):
            output = dcgrucell(output, state)
            hidden_states.append(output)
        return output, torch.stack(hidden_states)
