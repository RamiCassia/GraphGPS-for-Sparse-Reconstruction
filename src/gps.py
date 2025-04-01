import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from torch_geometric.nn.resolver import activation_resolver
from torch.nn import Dropout, Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from mamba_ssm import Mamba, Mamba2

from src.graph_sage_layer import SAGE_C
from src.graph_convolutional_layer import GCN_C
from src.graph_attention_layer import GAT_C
from src.model_components import ExphormerAttention, MemoryEfficientLinearAttention


class GPSConv(torch.nn.Module):

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = 'relu',
        att_type: str = 'MAMBA2',
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        headdim: int = 8,
        mp: str = 'GATMOD',
        kernel: str = 'exp',
        sage_aggr: str = 'max',
        act_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()

        self.channels = channels
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
        self.mp = mp
        self.act = nn.ReLU()
        self.kernel = kernel
        self.sage_aggr = sage_aggr

        if self.mp == 'GATMOD':
            self.convh1 = GAT_C(in_channels = self.channels, out_channels = self.channels//self.heads, heads = self.heads, bias = True, modded = True)
            self.convh4 = GAT_C(in_channels = self.channels, out_channels = self.channels//self.heads, heads = self.heads, bias = True, modded = True)
        elif self.mp == 'GAT':
            self.convh1 = GAT_C(in_channels = self.channels, out_channels = self.channels//self.heads, heads = self.heads, bias = True, modded = False)
            self.convh4 = GAT_C(in_channels = self.channels, out_channels = self.channels//self.heads, heads = self.heads, bias = True, modded = False)
        elif self.mp == 'SAGE':
            self.convh1 = SAGE_C(in_channels = self.channels, out_channels = self.channels, bias = True, aggr = self.sage_aggr)
            self.convh4 = SAGE_C(in_channels = self.channels, out_channels = self.channels, bias = True, aggr = self.sage_aggr)
        elif self.mp == 'GCN':
            self.convh1 = GCN_C(in_channels = self.channels, out_channels = self.channels, bias = True)
            self.convh4 = GCN_C(in_channels = self.channels, out_channels = self.channels, bias = True)


        self.mlp1 = Sequential(Linear(channels, channels*2, bias = False), activation_resolver(act, **(act_kwargs or {})), Dropout(dropout), Linear(channels*2, channels, bias = False), Dropout(dropout))
        self.mlp2 = Sequential(Linear(channels, channels*2, bias = False), activation_resolver(act, **(act_kwargs or {})), Dropout(dropout), Linear(channels*2, channels, bias = False), Dropout(dropout))
        self.mlpT = Sequential(Linear(channels, channels, bias = False), activation_resolver(act, **(act_kwargs or {})), Dropout(dropout), Linear(channels, channels, bias = False), Dropout(dropout))

        if self.att_type == 'TRANSFORMER':
            self.attn = torch.nn.MultiheadAttention(channels, headdim, dropout=attn_dropout, batch_first=True)

        if self.att_type == 'EXPHORMER':
            self.self_attn = ExphormerAttention(in_dim = channels, out_dim = channels, num_heads = headdim, bias=False, use_virt_nodes = False)

        if self.att_type == 'MAMBA2':
            self.self_attn = Mamba2(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand, headdim = headdim)  # make sure d_model * expand / headdim = multiple of 8

        if self.att_type == 'MAMBA':
            self.self_attn = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand = expand)

        if self.att_type == 'TRANSFORMERLIN':
            self.self_attn = MemoryEfficientLinearAttention(embed_dim = channels, num_heads = headdim, kernel_type = self.kernel)

        self.layernorm1 = nn.LayerNorm(channels)
        self.layernorm2 = nn.LayerNorm(channels)
        self.layernorm3 = nn.LayerNorm(channels)
        self.layernorm4 = nn.LayerNorm(channels)

    def reorder_graph(self, graph, Z_values, random = False): 

        perm = torch.randperm(graph.size(0)) if random else torch.argsort(Z_values)
        inverse_perm = torch.argsort(perm)
        graph = graph[perm]

        return graph, inverse_perm


    def forward(self, x, edge_index, bool_mask, batch, Z_values, random, exphormer_chars, epoch):

        hs = []

        x = self.mlp1(x)

        if self.mp == 'NONE':
            h_local = torch.tensor([0])
        else:
            h = self.convh1(x, edge_index, Z_values = Z_values)
            h_local = torch.tensor([0])
            h = self.layernorm3(h + x)
            hs.append(h)


        if self.att_type == 'TRANSFORMER':

            x1 = x.clone()
            h, att_mask = to_dense_batch(x, batch)
            h, _ = self.attn(h, h, h, key_padding_mask=~att_mask, need_weights=False)
            h = h[att_mask]

            h_att = torch.tensor([0])

            h = self.layernorm1(h + x1)
            x2 = h.clone()
            h = self.mlpT(h)
            h = self.layernorm2(h + x2)

            hs.append(h)


        if self.att_type == 'TRANSFORMERLIN':

            x1 = x.clone()
            h, att_mask = to_dense_batch(x, batch)
            h = self.self_attn(h)
            h = h[att_mask]

            h_att = torch.tensor([0])

            h = self.layernorm1(h + x1)
            x2 = h.clone()
            h = self.mlpT(h)
            h = self.layernorm2(h + x2)

            hs.append(h)


        if self.att_type == 'MAMBA' or self.att_type == 'MAMBA2':


            x1 = x.clone()
            x, inverse_perm = self.reorder_graph(x, Z_values, random)
            h, att_mask = to_dense_batch(x, batch)
            h = self.self_attn(h)[att_mask]
            h = h[inverse_perm]

            h_att = torch.tensor([0])

            h = self.layernorm1(h + x1)
            x2 = h.clone()
            h = self.mlpT(h)
            h = self.layernorm2(h + x2)

            hs.append(h)

        if self.att_type == 'EXPHORMER':

            x1 = x.clone()

            h = self.self_attn(h, exphormer_chars[0], exphormer_chars[1], exphormer_chars[2], exphormer_chars[3], exphormer_chars[4])

            h_att = torch.tensor([0]) 

            h = self.layernorm1(h + x1)
            x2 = h.clone()
            h = self.mlpT(h)
            h = self.layernorm2(h + x2)

            hs.append(h)

        if self.att_type == 'NONE':
            h_att = torch.tensor([0])

        out = sum(hs)
        global_cont = torch.norm(h)/torch.norm(out)
        out = self.layernorm4(out)

        if self.mp == 'NONE':
            out = self.mlp2(out)
        else:
            out = self.mlp2(out)
            out = self.act(out)
            out = self.convh4(out, edge_index, Z_values = Z_values)

        return out, global_cont, h_att, h_local
    
class GraphModel(torch.nn.Module):
    def __init__(self,
                 channels: int,
                 pe_dim_in: int,
                 pe_dim: int,
                 att_type: str,
                 d_state: int,
                 d_conv: int,
                 heads,
                 expand,
                 headdim,
                 edge_indices_list,
                 bool_mask_list,
                 shape,
                 mp,
                 exphormer_chars_list = None,
                 dense_graph = False,
                 masked_projection = True,
                 kernel = 'exp',
                 sage_aggr = 'max'):
        super().__init__()

        self.node_lin = Linear(10, channels - pe_dim)
        self.pe_dim = pe_dim
        if self.pe_dim > 0:
            self.pe_lin = Linear(pe_dim_in, pe_dim)
        self.edge_lin = Linear(1, channels)
        self.att_type = att_type
        self.edge_indices_list = edge_indices_list
        self.bool_mask_list = bool_mask_list
        self.shape = shape
        self.channels = channels
        self.batch = torch.zeros((self.shape[2] + 2)**3, dtype = torch.long).to('cuda')
        self.heads = heads
        self.mp = mp
        self.exphormer_chars_list = exphormer_chars_list
        self.masked_projection = masked_projection
        self.dense_graph = dense_graph
        self.kernel = kernel
        self.sage_aggr = sage_aggr

        if self.mp == 'GATMOD':
            self.convh = GAT_C(in_channels = self.channels, out_channels = self.channels//self.heads, heads = self.heads, bias = True, modded = True)
            self.convo = GAT_C(in_channels = self.channels, out_channels = 10//1, heads = 1, bias = True, modded = True)
        elif self.mp == 'GAT':
            self.convh = GAT_C(in_channels = self.channels, out_channels = self.channels//self.heads, heads = self.heads, bias = True, modded = False)
            self.convo = GAT_C(in_channels = self.channels, out_channels = 10//1, heads = 1, bias = True, modded = False)
        elif self.mp == 'SAGE':
            self.convh = SAGE_C(in_channels = self.channels, out_channels = self.channels, bias = True, aggr = self.sage_aggr)
            self.convo = SAGE_C(in_channels = self.channels, out_channels = 10, bias = True, aggr = self.sage_aggr)
        elif self.mp == 'GCN':
            self.convh = GCN_C(in_channels = self.channels, out_channels = self.channels, bias = True)
            self.convo = GCN_C(in_channels = self.channels, out_channels = 10, bias = True)
        elif self.mp == 'NONE':
            self.mlp_out = Sequential(Linear(channels, channels // 2), ReLU(), Linear(channels // 2, channels//4), ReLU(), Linear(channels // 4, 10))

        self.convs = nn.ModuleList()
        for i in range(len(edge_indices_list)):
            self.convs.append(GPSConv(channels = self.channels, heads=self.heads, attn_dropout=0.5, att_type= self.att_type, d_state=d_state, d_conv=d_conv, expand = expand, headdim = headdim, mp = self.mp, kernel = self.kernel, sage_aggr = self.sage_aggr))


    def flatten(self, x):
        t, c, h, w, d = x.shape
        x = x.reshape(t * c, h * w * d).T
        return x

    def reshape_back_to_original(self, flat_data, t, c, h, w, d):

        reshaped_data = flat_data.T.reshape(t * c, h, w, d)
        original_data = reshaped_data.reshape((t, c, h, w, d))

        return original_data

    def replication_pad_3d(self, x, pad = (1,1,1,1,1,1)):
        x = F.pad(x, pad, mode='replicate')
        return x

    def enforce_bc(self, x, t, c, h, w, d):

        x = self.reshape_back_to_original(x, t, c, h, w, d)
        x = x[:,:,1:-1, 1:-1, 1:-1]
        x = self.replication_pad_3d(x)
        x = self.flatten(x)
        return x

    def forward(self, x, pe, Z_values, epoch, epochs_to_order):

        global_conts = []
        h_att_list = []
        h_local_list = []

        if epoch < epochs_to_order:
            random = True
        else:
            random = False

        t, c, h, w, d = self.shape
        x_pe = pe.clone()


        if self.masked_projection == True:
            x_sq = torch.zeros(((h+2)*(w+2)*(d+2), self.channels - self.pe_dim)).squeeze(-1).cuda()
            x_mask = x[self.bool_mask_list[0]]
            x_mask = self.node_lin(x_mask)
            x_sq[self.bool_mask_list[0]] = x_mask
            if self.pe_dim > 0:
                x = torch.cat((x_sq, self.pe_lin(x_pe)), 1)
            else:
                x = x_sq    
        else:
            x = torch.cat((self.node_lin(x.squeeze(-1)), self.pe_lin(x_pe)), 1)

        for i, layer in enumerate(self.convs):

            if self.dense_graph == True:
                i = -1

            x, global_cont, h_att, h_local = layer(x, self.edge_indices_list[i], self.bool_mask_list[i], self.batch, Z_values, random, self.exphormer_chars_list[i], epoch)
            x = self.enforce_bc(x, t, int(self.channels/t), h + 2, w + 2, d + 2)
            global_conts.append(global_cont.cpu().detach().numpy())
            h_att_list.append(h_att.cpu().detach().numpy())
            h_local_list.append(h_local.cpu().detach().numpy())

        if self.mp == 'NONE':
            x = self.mlp_out(x)
        else:
            x = self.convh(x, self.edge_indices_list[-1], Z_values = Z_values)
            x = self.convo(x, self.edge_indices_list[-1], Z_values = Z_values)


        x = self.reshape_back_to_original(x, t, 5, h + 2, w + 2, d + 2)
        x = x[:,:,1:-1,1:-1,1:-1].clone()
        xp = x.clone()
        xp[:, [0, 4], :, :, :] = torch.abs((x[:, [0, 4], :, :, :].clone()))

        return xp, global_conts, h_att_list, h_local_list
