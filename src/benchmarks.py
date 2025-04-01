import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

class CNN_BM(nn.Module):

    def __init__(self):
        super(CNN_BM, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv6 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv7 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv8 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv9 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv10 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv11 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.conv12 = nn.Conv3d(in_channels=16, out_channels=5, kernel_size=3, stride=1, padding=1, padding_mode = 'replicate')
        self.act = nn.ReLU()

    def forward(self, x):

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        xr = x.clone()
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.act(self.conv7(x))
        x = self.act(self.conv8(x))
        x = self.act(self.conv9(x))
        x = x + xr
        x = self.act(self.conv10(x))
        x = self.act(self.conv11(x))
        x = self.conv12(x)

        xp = x.clone()
        xp[:, [0, 4], :, :, :] = torch.abs((x[:, [0, 4], :, :, :].clone()))

        return xp


class GNN_BM(torch.nn.Module):
    def __init__(self, channels, edge_indices_list, shape, heads):
        super(GNN_BM, self).__init__()

        self.heads = heads

        self.convi = gnn.GATConv(10, channels//self.heads, heads = self.heads)
        self.convhi = gnn.GATConv(channels, channels//self.heads, heads = self.heads)
        self.convho = gnn.GATConv(channels, channels//self.heads, heads = self.heads)
        self.convo = gnn.GATConv(channels, 10, heads = 1)

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for i in range(1, len(edge_indices_list) - 1):
            self.convs1.append(gnn.GATConv(channels, channels // self.heads, heads=self.heads))
            self.convs2.append(gnn.GATConv(channels, channels // self.heads, heads=self.heads))

        self.hidden_channels = channels

        self.edge_indices_list = edge_indices_list

        self.act = nn.ReLU()

        self.shape = shape

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

    def forward(self, x):

        t, c, h, w, d = self.shape

        x = self.convi(x, self.edge_indices_list[-1])
        x = self.act(x)

        x = self.enforce_bc(x, t, int(self.hidden_channels/t), h + 2, w + 2, d + 2)
        x = self.convhi(x, self.edge_indices_list[-1])
        x = self.act(x)

        for i in range(1, len(self.edge_indices_list) - 1):

            x = self.enforce_bc(x, t, int(self.hidden_channels/t), h + 2, w + 2, d + 2)
            x = self.convs1[i - 1](x, self.edge_indices_list[-1])
            x = self.act(x)

            x = self.enforce_bc(x, t, int(self.hidden_channels/t), h + 2, w + 2, d + 2)
            x = self.convs2[i - 1](x, self.edge_indices_list[-1])
            x = self.act(x)

        x = self.enforce_bc(x, t, int(self.hidden_channels/t), h + 2, w + 2, d + 2)
        x = self.convho(x, self.edge_indices_list[-1])
        x = self.act(x)

        x = self.enforce_bc(x, t, int(self.hidden_channels/t), h + 2, w + 2, d + 2)
        x = self.convo(x, self.edge_indices_list[-1])

        x = self.reshape_back_to_original(x, t, int(10/t), h + 2, w + 2, d + 2)

        x = x[:,:,1:-1,1:-1,1:-1].clone()
        xp = x.clone()
        xp[:, [0, 4], :, :, :] = torch.abs((x[:, [0, 4], :, :, :].clone()))

        return xp


class AE_BM(nn.Module):
    def __init__(self, channels, num_layers, shape):
        super(AE_BM, self).__init__()

        self.bottleneck_dim = int(channels/(2**(num_layers-1)))
        
        encoder_dims = [channels] + [max(self.bottleneck_dim, channels // (2 ** (i + 1))) for i in range(num_layers)]
        decoder_dims = encoder_dims[::-1] 

        self.encoder = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            self.encoder.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))

        self.decoder = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.decoder.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))

        self.lineari1 = nn.Linear(10, 512)

        self.linearo1 = nn.Linear(512, 10)
        
        self.act = nn.ReLU()

        self.shape = shape
        self.hidden_channels = channels
   
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
    
    
    def forward(self, x):

        t, c, h, w, d = self.shape

        x = self.lineari1(x)
        x = self.act(x)

        for i, layer in enumerate(self.encoder):

            x = self.enforce_bc(x, t, int(self.hidden_channels/t/(2**(i))), h + 2, w + 2, d + 2)
            x = layer(x)
            x = self.act(x)

        for i, layer in enumerate(self.decoder):
 
            if i == 0:
                x = self.enforce_bc(x, t, int((self.bottleneck_dim/t)*(2**(i))), h + 2, w + 2, d + 2)
            else:
                x = self.enforce_bc(x, t, int((self.bottleneck_dim/t)*(2**(i-1))), h + 2, w + 2, d + 2)
       
            x = layer(x)
            x = self.act(x)

        x = self.linearo1(x)

        x = self.reshape_back_to_original(x, t, int(10/t), h + 2, w + 2, d + 2)

        x = x[:,:,1:-1,1:-1,1:-1].clone()
        xp = x.clone()
        xp[:, [0, 4], :, :, :] = torch.abs((x[:, [0, 4], :, :, :].clone()))

        return x
