import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

import numpy as np
import torch
import torch.nn as nn

class Sinusoidal_pe(nn.Module):
    def __init__(self, channels, dtype_override=None):

        super(Sinusoidal_pe, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels

    def get_emb(self, sin_inp):

        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor, nx, ny, nz):

        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        tensor = torch.permute(tensor, (0,3,2,4,1))

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x.cuda(), self.inv_freq.cuda())
        sin_inp_y = torch.einsum("i,j->ij", pos_y.cuda(), self.inv_freq.cuda())
        sin_inp_z = torch.einsum("i,j->ij", pos_z.cuda(), self.inv_freq.cuda())
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = self.get_emb(sin_inp_y).unsqueeze(1)
        emb_z = self.get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)

        self.cached_penc = torch.permute(self.cached_penc, (0,4,2,1,3))
        self.cached_penc = torch.reshape(self.cached_penc, (10, ny*nx*nz))
        self.cached_penc = self.cached_penc.T


        return self.cached_penc
