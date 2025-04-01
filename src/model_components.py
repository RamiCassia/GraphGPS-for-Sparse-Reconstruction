import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np

class MemoryEfficientLinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_type="exp"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        
        self.kernel_type = kernel_type

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def kernel_function(self, x):
        if self.kernel_type == "exp":
            return torch.exp(x - x.max(dim=-1, keepdim=True).values) 
        elif self.kernel_type == "gaussian":
            norm_x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)
            return torch.exp(-torch.norm(norm_x, dim=-1, keepdim=True) ** 2)
        elif self.kernel_type == 'elup1':
            return torch.nn.functional.elu(x) + 1

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q_kern = self.kernel_function(q)
        k_kern = self.kernel_function(k)

        kv = torch.einsum("bhsd,bhse->bhde", k_kern, v)  

        denom = torch.einsum("bhqd,bhsd->bhq", q_kern, k_kern) + 1e-6  

        y = torch.einsum("bhqd,bhde->bhqe", q_kern, kv) / denom.unsqueeze(-1) 

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(y)


class ExphormerAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, bias, use_virt_nodes=False):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not dividable by the number of heads')

        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.bias = bias

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=bias)
        self.E = nn.Linear(1, self.out_dim * num_heads, bias=bias)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=bias)


    def propagate_attention(self, K_h, Q_h, V_h, E, edge_index):

        src = K_h[edge_index[0].to(torch.long)]
        dest = Q_h[edge_index[1].to(torch.long)]
        score = torch.mul(src, dest)

        score = score / np.sqrt(self.out_dim)

        score = torch.mul(score, E)
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))

        msg = V_h[edge_index[0].to(torch.long)] * score
        wV = torch.zeros_like(V_h)
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        Z = score.new_zeros(V_h.size(0), self.num_heads, 1)
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

        return wV, Z

    def forward(self, x, expander_edge_index, expander_edge_attr, virt_h, virt_edge_index, virt_edge_attr):

        edge_attr = expander_edge_attr
        edge_index = expander_edge_index
        h = x
        num_node = x.size(0)

        if self.use_virt_nodes:

            h = torch.cat([h, virt_h], dim=0)
            edge_index = torch.cat([edge_index, virt_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr, virt_edge_attr], dim=0)

        Q_h = self.Q(h)
        K_h = self.K(h)
        E = self.E(edge_attr)
        V_h = self.V(h)

        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        E = E.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)

        wV, Z = self.propagate_attention(K_h, Q_h, V_h, E, edge_index)

        h_out = wV / (Z + 1e-6)

        h_out = h_out.view(-1, self.out_dim * self.num_heads)

        virt_h = h_out[num_node:]
        h_out = h_out[:num_node]

        return h_out
