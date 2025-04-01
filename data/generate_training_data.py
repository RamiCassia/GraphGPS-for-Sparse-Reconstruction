import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy as sc
from scipy.interpolate import griddata, RBFInterpolator


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from itertools import product
from torch_geometric.data import Data
import networkx as nx


class Utils():

    def get_voxels_random(field, probability, zero = False):

        shape = np.shape(field)[2:5]

        np.random.seed(42)
        rand_values = np.random.uniform(0,1,shape)
        mask = (rand_values <= probability)
        field_s = field.copy()

        expanded_mask = np.expand_dims(mask, axis=(0, 1))
        masked_array = np.where(expanded_mask, field, np.nan)
        mean = np.nanmean(masked_array, axis=(0,2,3,4))

        for n in range(0, np.shape(field)[1]):

            field_s[0,n][mask] = mean[n] if not zero else 0
            field_s[1,n][mask] = mean[n] if not zero else 0

        return mask, field_s

    def to_networkx(data, directed=True):
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        edge_index = data.edge_index.cpu().numpy()
        x = data.x.cpu().numpy()

        for i, features in enumerate(x):
            G.add_node(i, features=features)

        for i, j in edge_index.T:
            G.add_edge(i, j)

        return G

class Plot():

    def plot_unfolded(field, field_s, title = '', channel = 0, time = 0, save = False, epoch = 0, path = ''):

        n = np.shape(field)[2]
        fig = plt.figure(figsize=(27, 9))
        gs = GridSpec(3, 9, figure=fig, wspace=0.05, hspace=0.05)
        omit_subplots = [(0, 0), (0, 2), (0, 3), (2,0), (2,2), (2,3)]

        def plot_single(offset, f):
            for i in range(3):
                for j in range(4):
                    if (i, j) not in omit_subplots:
                        ax = fig.add_subplot(gs[i, j + offset])

                        if (i,j) == (1,0):
                            ax.imshow(np.fliplr(np.flipud(f[time,channel,0,:,:])))
                        if (i,j) == (1,1):
                            ax.imshow(np.flipud(f[time,channel,:,:,0]))
                        if (i,j) == (1,2):
                            ax.imshow(np.flipud(f[time,channel,n-1,:,:]))
                        if (i,j) == (1,3):
                            ax.imshow(np.fliplr(np.flipud(f[time,channel,:,:,n-1])))
                        if (i,j) == (0,1):
                            ax.imshow(np.flipud(f[time,channel,:,n-1,:]))
                        if (i,j) == (2,1):
                            ax.imshow(f[time,channel,:,0,:])

                        ax.set_xticks([])
                        ax.set_yticks([])

        plot_single(offset = 0, f = field)
        plot_single(offset = 5, f = field_s)

        fig.suptitle(title, fontsize=16)

        if save:
            plt.savefig(path + str(epoch) + '.png',  bbox_inches='tight')

        plt.show()

    def visualize_cross_section(G, height, width, depth, section_depth, bool_mask=None):

        print(bool_mask[:,:,0])

        pos = {}
        for h, w, d in product(range(height), range(width), range(depth)):
            node_index = np.ravel_multi_index((h, w, d), (height, width, depth))
            if d == section_depth:
                pos[node_index] = (w, -h) 

        nodes = [node for node in G.nodes() if node in pos]
        edges = [(i, j) for i, j in G.edges() if i in pos and j in pos]

        plt.figure(figsize=(10, 10))
        nx.draw(G, pos=pos, nodelist=nodes, edgelist=edges, with_labels=False, node_size=600, node_color="blue", edge_color="gray", width = 4, arrowsize = 20)

        if bool_mask is not None:
            true_nodes = [np.ravel_multi_index((h, w, section_depth), (height, width, depth)) for h, w in product(range(height), range(width)) if bool_mask[h, w, section_depth]]
            nx.draw_networkx_nodes(G, pos, nodelist=true_nodes, node_color='red', node_size=600, node_shape = 's')
        plt.show()

    def visualize_3d_graph(G, height, width, depth, bool_mask=None):
        pos = {}
        for h, w, d in product(range(height), range(width), range(depth)):
            node_index = np.ravel_multi_index((h, w, d), (height, width, depth))
            pos[node_index] = (w, h, d)  

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for node, (x, y, z) in pos.items():
            color = 'red' if bool_mask is not None and bool_mask[y, x, z] else 'blue'
            ax.scatter(x, y, z, color=color, s=50)

        for i, j in G.edges():
            x = [pos[i][0], pos[j][0]]
            y = [pos[i][1], pos[j][1]]
            z = [pos[i][2], pos[j][2]]
            ax.plot(x, y, z, color='gray')

        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Depth')
        plt.title("3D Graph Visualization")
        plt.show()

class GraphP3():

    def __init__(self, data, bool_mask, plot):
        super(GraphP3, self).__init__()
        self.data = data
        self.bool_mask = bool_mask
        self.plot = plot

    def create_directed_graph_from_5d_array(self):
        time, channel, height, width, depth = self.data.shape
        num_features = time * channel
        num_nodes = height * width * depth

        features = self.data.view(time * channel, height * width * depth).T

        edges = []
        for h, w, d in product(range(height), range(width), range(depth)):
            if not self.bool_mask[h, w, d]:
                continue
            index = torch.tensor(h * width * depth + w * depth + d, dtype=torch.long)
            neighbor_indices = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    for k in [-1, 0, 1]:
                        if i == 0 and j == 0 and k == 0:
                            continue
                        nh, nw, nd = h + i, w + j, d + k
                        if 0 <= nh < height and 0 <= nw < width and 0 <= nd < depth:
                            neighbor_index = torch.tensor(nh * width * depth + nw * depth + nd, dtype=torch.long)
                            neighbor_indices.append(neighbor_index)
            for neighbor_index in neighbor_indices:
                edges.append(torch.tensor([index, neighbor_index]))

        edge_index = torch.stack(edges, dim=1)
        x = torch.tensor(features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index)

    def add_masked_self_loops(x, edge_index, mask):

        num_nodes = x.shape[2]*x.shape[3]*x.shape[4]
        H, W, D = mask.shape

        if H * W * D != num_nodes:
            raise ValueError("Mask size (H*W*D) must equal the number of nodes.")

        flat_mask = mask.view(-1)
        self_loop_edge_index = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)

        masked_self_loop_edge_index = self_loop_edge_index[:, flat_mask]

        if edge_index.nelement() > 0:
            if masked_self_loop_edge_index.nelement() > 0:
                new_edge_index = torch.cat([edge_index, masked_self_loop_edge_index], dim=1)
            else:
                new_edge_index = edge_index
        else:
            new_edge_index = masked_self_loop_edge_index

        return new_edge_index

    def propagate_and_update(self):
        height, width, depth = self.bool_mask.shape

        bool_mask_list = []
        edge_indices_list = []

        while not self.bool_mask.all():
            graph_data = self.create_directed_graph_from_5d_array()

            if self.plot == True:
                G = Utils.to_networkx(graph_data, directed=True)
                Plot.visualize_cross_section(G, height, width, depth, section_depth=0, bool_mask=self.bool_mask)
                Plot.visualize_3d_graph(G, height, width, depth, bool_mask=self.bool_mask)

            bool_mask_list.append(self.bool_mask)
            edge_indices_list.append(graph_data.edge_index)

            new_bool_mask = self.bool_mask.clone()
            for h in range(height):
                for w in range(width):
                    for d in range(depth):
                        if self.bool_mask[h, w, d]:
                            for i in [-1, 0, 1]:
                                for j in [-1, 0, 1]:
                                    for k in [-1, 0, 1]:
                                        if i == 0 and j == 0 and k == 0:
                                            continue
                                        nh, nw, nd = h + i, w + j, d + k
                                        if 0 <= nh < height and 0 <= nw < width and 0 <= nd < depth:
                                            new_bool_mask[nh, nw, nd] = True

            self.bool_mask = new_bool_mask

        bool_mask_list.append(self.bool_mask)
        graph_data = self.create_directed_graph_from_5d_array()
        edge_indices_list.append(graph_data.edge_index)

        if self.plot == True:
            G = Utils.to_networkx(graph_data, directed=True)
            Plot.visualize_cross_section(G, height, width, depth, section_depth=0, bool_mask=self.bool_mask)
            Plot.visualize_3d_graph(G, height, width, depth, bool_mask=self.bool_mask)

        return graph_data.x, edge_indices_list, bool_mask_list

class GraphP2():

    def __init__(self, data, bool_mask, plot):
        super(GraphP2, self).__init__()
        self.data = data
        self.bool_mask = bool_mask
        self.plot = plot

    def create_directed_graph_from_5d_array(self):
        time, channel, height, width, depth = self.data.shape
        num_features = time * channel
        num_nodes = height * width * depth

        features = self.data.view(time * channel, height * width * depth).T

        edges = []
        for h, w, d in product(range(height), range(width), range(depth)):
            if not self.bool_mask[h, w, d]:
                continue
            index = torch.tensor(h * width * depth + w * depth + d, dtype=torch.long)
            neighbor_indices = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    for k in [-1, 0, 1]:
                        if i == 0 and j == 0 and k == 0:
                            continue
            
                        if abs(i) + abs(j) + abs(k) <= 2:
                            nh, nw, nd = h + i, w + j, d + k
                            if 0 <= nh < height and 0 <= nw < width and 0 <= nd < depth:
                                neighbor_index = torch.tensor(nh * width * depth + nw * depth + nd, dtype=torch.long)
                                neighbor_indices.append(neighbor_index)
            for neighbor_index in neighbor_indices:
                edges.append(torch.tensor([index, neighbor_index]))

        edge_index = torch.stack(edges, dim=1)
        x = torch.tensor(features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index)

    def add_masked_self_loops(x, edge_index, mask):

        num_nodes = x.shape[2]*x.shape[3]*x.shape[4]
        H, W, D = mask.shape

        if H * W * D != num_nodes:
            raise ValueError("Mask size (H*W*D) must equal the number of nodes.")

        flat_mask = mask.view(-1)
        self_loop_edge_index = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)

        masked_self_loop_edge_index = self_loop_edge_index[:, flat_mask]

        if edge_index.nelement() > 0:
            if masked_self_loop_edge_index.nelement() > 0:
                new_edge_index = torch.cat([edge_index, masked_self_loop_edge_index], dim=1)
            else:
                new_edge_index = edge_index
        else:
            new_edge_index = masked_self_loop_edge_index

        return new_edge_index

    def propagate_and_update(self):
        height, width, depth = self.bool_mask.shape

        bool_mask_list = []
        edge_indices_list = []

        while not self.bool_mask.all():
            graph_data = self.create_directed_graph_from_5d_array()

            if self.plot == True:
                G = Utils.to_networkx(graph_data, directed=True)
                Plot.visualize_cross_section(G, height, width, depth, section_depth=0, bool_mask=self.bool_mask)
                Plot.visualize_3d_graph(G, height, width, depth, bool_mask=self.bool_mask)

            bool_mask_list.append(self.bool_mask)
            edge_indices_list.append(graph_data.edge_index)

            new_bool_mask = self.bool_mask.clone()
            for h in range(height):
                for w in range(width):
                    for d in range(depth):
                        if self.bool_mask[h, w, d]:
                            for i in [-1, 0, 1]:
                                for j in [-1, 0, 1]:
                                    for k in [-1, 0, 1]:
                                        if i == 0 and j == 0 and k == 0:
                                            continue
                    
                                        if abs(i) + abs(j) + abs(k) <= 2:
                                            nh, nw, nd = h + i, w + j, d + k
                                            if 0 <= nh < height and 0 <= nw < width and 0 <= nd < depth:
                                                new_bool_mask[nh, nw, nd] = True

            self.bool_mask = new_bool_mask

        bool_mask_list.append(self.bool_mask)
        graph_data = self.create_directed_graph_from_5d_array()
        edge_indices_list.append(graph_data.edge_index)

        if self.plot == True:
            G = Utils.to_networkx(graph_data, directed=True)
            Plot.visualize_cross_section(G, height, width, depth, section_depth=0, bool_mask=self.bool_mask)
            Plot.visualize_3d_graph(G, height, width, depth, bool_mask=self.bool_mask)

        return graph_data.x, edge_indices_list, bool_mask_list


class GraphP1():

    def __init__(self, data, bool_mask, plot):
        super(GraphP1, self).__init__()
        self.data = data
        self.bool_mask = bool_mask
        self.plot = plot

    def create_directed_graph_from_5d_array(self):
        time, channel, height, width, depth = self.data.shape
        num_features = time * channel
        num_nodes = height * width * depth

        features = self.data.view(time * channel, height * width * depth).T

        edges = []
        for h, w, d in product(range(height), range(width), range(depth)):
            if not self.bool_mask[h, w, d]:
                continue
            index = torch.tensor(h * width * depth + w * depth + d, dtype=torch.long)
            neighbor_indices = []
            for i, j, k in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                nh, nw, nd = h + i, w + j, d + k
                if 0 <= nh < height and 0 <= nw < width and 0 <= nd < depth:
                    neighbor_index = torch.tensor(nh * width * depth + nw * depth + nd, dtype=torch.long)
                    neighbor_indices.append(neighbor_index)
            for neighbor_index in neighbor_indices:
                edges.append(torch.tensor([index, neighbor_index]))

        edge_index = torch.stack(edges, dim=1)
        x = torch.tensor(features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index)

    def add_masked_self_loops(x, edge_index, mask):

        num_nodes = x.shape[2]*x.shape[3]*x.shape[4]
        H, W, D = mask.shape

        if H * W * D != num_nodes:
            raise ValueError("Mask size (H*W*D) must equal the number of nodes.")

        flat_mask = mask.view(-1)
        self_loop_edge_index = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)

        masked_self_loop_edge_index = self_loop_edge_index[:, flat_mask]

        if edge_index.nelement() > 0:
            if masked_self_loop_edge_index.nelement() > 0:
                new_edge_index = torch.cat([edge_index, masked_self_loop_edge_index], dim=1)
            else:
                new_edge_index = edge_index
        else:
            new_edge_index = masked_self_loop_edge_index

        return new_edge_index

    def propagate_and_update(self):
        height, width, depth = self.bool_mask.shape

        bool_mask_list = []
        edge_indices_list = []

        while not self.bool_mask.all():
            graph_data = self.create_directed_graph_from_5d_array()

            if self.plot == True:
                G = Utils.to_networkx(graph_data, directed=True)
                Plot.visualize_cross_section(G, height, width, depth, section_depth=0, bool_mask=self.bool_mask)
                Plot.visualize_3d_graph(G, height, width, depth, bool_mask=self.bool_mask)

            bool_mask_list.append(self.bool_mask)
            edge_indices_list.append(graph_data.edge_index)

            new_bool_mask = self.bool_mask.clone()
            for h in range(height):
                for w in range(width):
                    for d in range(depth):
                        if self.bool_mask[h, w, d]:
                            for i, j, k in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                                nh, nw, nd = h + i, w + j, d + k
                                if 0 <= nh < height and 0 <= nw < width and 0 <= nd < depth:
                                    new_bool_mask[nh, nw, nd] = True

            self.bool_mask = new_bool_mask

        bool_mask_list.append(self.bool_mask)
        graph_data = self.create_directed_graph_from_5d_array()
        edge_indices_list.append(graph_data.edge_index)

        if self.plot == True:
            G = Utils.to_networkx(graph_data, directed=True)
            Plot.visualize_cross_section(G, height, width, depth, section_depth=0, bool_mask=self.bool_mask)
            Plot.visualize_3d_graph(G, height, width, depth, bool_mask=self.bool_mask)

        return graph_data.x, edge_indices_list, bool_mask_list

nxyz = 64
dt = 0.0005
random = 90
tend = 0.3

for con in [1,2,3,4,5,6]: 
    for p in [3,2,1]:

        data_path = '/path/'
        data_sparsity = 'random_' + str(random) + '_con_' + str(con) + '_nxyz_' + str(nxyz) + '_dt_' + str(dt) + '_tend_' + str(tend) + '_p_' + str(p)

        os.makedirs(data_path + data_sparsity, exist_ok=True)

        field = np.load(data_path + 'field_con_' + str(con) + '_nxyz_' + str(nxyz) + '_dt_' + str(dt) + '_tend_' + str(tend) + '.npy')
        mask, field_s = Utils.get_voxels_random(field, random/100, zero = False) 
        Plot.plot_unfolded(field_s, field)

        mask = np.pad(mask, pad_width=((1, 1), (1, 1), (1, 1)), mode='edge')
        field_s = np.pad(field_s, pad_width=((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)), mode='edge')
        field = torch.tensor(field, dtype=torch.float32)
        field_s = torch.tensor(field_s, dtype=torch.float32)
        mask_data = torch.tensor(mask)

        if p==3:
            _, edge_indices_list, bool_mask_list  = GraphP3(field_s, ~mask_data, plot=False).propagate_and_update()
        elif p==2:
            _, edge_indices_list, bool_mask_list  = GraphP2(field_s, ~mask_data, plot=False).propagate_and_update()
        elif p==1:
            _, edge_indices_list, bool_mask_list  = GraphP1(field_s, ~mask_data, plot=False).propagate_and_update()

        torch.save(field, data_path + data_sparsity + '/field.pt')
        torch.save(field_s, data_path + data_sparsity + '/field_s.pt')
        torch.save(mask_data, data_path + data_sparsity + '/mask.pt')
        torch.save(edge_indices_list, data_path + data_sparsity + '/edge_list.pt')
        torch.save(bool_mask_list, data_path + data_sparsity + '/bool_list.pt')


