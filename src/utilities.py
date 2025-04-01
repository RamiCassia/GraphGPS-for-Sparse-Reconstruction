import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.transforms as T

class Utils():

    def get_grad_rho_norm(field):

        N = field.shape[2]

        delta = 1/(N-1)

        drho_dy = (field[0, 0, 2:, 1:-1, 1:-1] - field[0, 0, :-2, 1:-1, 1:-1])/(2*delta)
        drho_dx = (field[0, 0, 1:-1, 2:, 1:-1] - field[0, 0, 1:-1, :-2, 1:-1])/(2*delta)
        drho_dz = (field[0, 0, 1:-1, 1:-1, 2:] - field[0, 0, 1:-1, 1:-1, :-2])/(2*delta)

        grad_rho_norm = torch.sqrt(drho_dx**2 + drho_dy**2 + drho_dz**2)

        grad_rho_norm_min = grad_rho_norm.min()
        grad_rho_norm_max = grad_rho_norm.max()
        grad_rho_norm = (grad_rho_norm - grad_rho_norm_min) / (grad_rho_norm_max - grad_rho_norm_min)

        grad_rho_norm = F.pad(grad_rho_norm.unsqueeze(0), pad=(1, 1, 1, 1, 1, 1), mode='replicate')
        grad_rho_norm = grad_rho_norm.squeeze(0)

        return grad_rho_norm.reshape((N)**3)


    def get_grad_p_norm(field):

        N = field.shape[2]

        delta = 1/(N-1)

        p = field[:,0:1]*(1.4-1)*(field[:,4:5] - 0.5*(field[:,1:2]**2 + field[:,2:3]**2 + field[:,3:4]**2))

        dp_dy = (p[0, 0, 2:, 1:-1, 1:-1] - p[0, 0, :-2, 1:-1, 1:-1])/(2*delta)
        dp_dx = (p[0, 0, 1:-1, 2:, 1:-1] - p[0, 0, 1:-1, :-2, 1:-1])/(2*delta)
        dp_dz = (p[0, 0, 1:-1, 1:-1, 2:] - p[0, 0, 1:-1, 1:-1, :-2])/(2*delta)

        grad_p_norm = torch.sqrt(dp_dx**2 + dp_dy**2 + dp_dz**2)

        grad_p_norm_min = grad_p_norm.min()
        grad_p_norm_max = grad_p_norm.max()
        grad_p_norm = (grad_p_norm - grad_p_norm_min) / (grad_p_norm_max - grad_p_norm_min)

        grad_p_norm = F.pad(grad_p_norm.unsqueeze(0), pad=(1, 1, 1, 1, 1, 1), mode='replicate')
        grad_p_norm = grad_p_norm.squeeze(0)


        return grad_p_norm.reshape((N)**3)


    def get_div_v(field):

            N = field.shape[2]

            delta = 1/(N-1)

            dv_dy = (field[0, 2, 2:, 1:-1, 1:-1] - field[0, 2, :-2, 1:-1, 1:-1])/(2*delta)
            du_dx = (field[0, 1, 1:-1, 2:, 1:-1] - field[0, 1, 1:-1, :-2, 1:-1])/(2*delta)
            dw_dz = (field[0, 3, 1:-1, 1:-1, 2:] - field[0, 3, 1:-1, 1:-1, :-2])/(2*delta)

            div_v = dv_dy + du_dx + dw_dz

            div_v = -div_v
            div_v = torch.max(div_v, torch.zeros_like(div_v))

            div_v_min = div_v.min()
            div_v_max = div_v.max()
            div_v = (div_v - div_v_min) / (div_v_max - div_v_min)

            div_v = F.pad(div_v.unsqueeze(0), pad=(1, 1, 1, 1, 1, 1), mode='replicate')
            div_v = div_v.squeeze(0)


            return div_v.reshape((N)**3)


    def get_Z(field):

        grad_rho = Utils.get_grad_rho_norm(field)
        grad_p = Utils.get_grad_p_norm(field)
        div_v = Utils.get_div_v(field)

        return torch.vstack([grad_rho, grad_p, div_v]).T

    def euclid_pe(n_dim, min = 0, max = 1):

        x = np.linspace(min, max, n_dim)
        y = np.linspace(min, max, n_dim)
        z = np.linspace(min, max, n_dim)
        X, Y, Z = np.meshgrid(x, y, z)
        node_positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        graph_center = np.mean(node_positions, axis=0)
        distances = np.linalg.norm(node_positions - graph_center, axis=1)
        max_distance = np.max(distances)
        normalized_distances = distances / max_distance

        return normalized_distances.reshape(n_dim**3, 1)

    def random_walk_pe(x, edge_index, walk_length):

        transform = T.AddRandomWalkPE(walk_length=walk_length)
        data = Data(x = x.T, edge_index=edge_index)
        data = transform(data)
        return data.random_walk_pe


    def laplacePE(x, edge_index, k):

        transform = T.AddLaplacianEigenvectorPE(k, is_undirected=True)
        data = Data(x=x.T, edge_index=edge_index)
        data = transform(data)

        return data.laplacian_eigenvector_pe


    def save_checkpoint(model, optimizer, scheduler, save_dir):
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, save_dir)

    def load_checkpoint(model, optimizer, scheduler, load_dir):

        checkpoint = torch.load(load_dir)
        model.load_state_dict(checkpoint['model_state_dict'])

        if (not optimizer is None):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Pretrained model loaded!')

        return model, optimizer, scheduler

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

 
    def flatten(x):
        t, c, h, w, d = x.shape
        x = x.reshape(t * c, h * w * d).T
        return x

    def create_project(result_path, run_name):

        path_fig = result_path + run_name + '/figures/'
        path_model = result_path + run_name + '/models/'
        path_output = result_path + run_name + '/outputs/'
        path_stats = result_path + run_name + '/stats/'
        path_parameters = result_path + run_name + '/parameters/'

        if os.path.exists(result_path) == False:
            os.mkdir(result_path)

        if os.path.exists(result_path + run_name + '/') == False:
            os.mkdir(result_path + run_name + '/')

        if os.path.exists(path_fig) == False:
            os.mkdir(path_fig)
        if os.path.exists(path_model) == False:
            os.mkdir(path_model)
        if os.path.exists(path_output) == False:
            os.mkdir(path_output)
        if os.path.exists(path_stats) == False:
            os.mkdir(path_stats)
        if os.path.exists(path_parameters) == False:
            os.mkdir(path_parameters)

        return path_fig, path_model, path_output, path_stats, path_parameters


    def dict_to_txt(path_parameters, param_dict):

        indent=""
        f = open(path_parameters + 'params.txt', "w")
        f.write("\n")

        for key, value in param_dict.items():
            f.write(f"{indent}{key}: ")

            if isinstance(value, list):
                i = 0
                first_item = True
                for sublist in value:
                    if first_item:
                        first_item = False
                    else:
                        f.write(" " * len(key) + "  ")

                    if isinstance(sublist, list):
                        f.write(f"{sublist}\n")
                        i += 1

                        if i == len(value):
                            f.write("\n")
                    else:
                        f.write(f"{sublist}\n")
            else:
                f.write(f"{value}\n\n")


    def change_filler_value(field, mask, value):

        mask = mask.bool()

        expanded_mask = mask.expand(field.shape[0], field.shape[1], -1, -1, -1)

        masked_field = field.masked_fill(~expanded_mask, value)

        return masked_field


    def feature_propagation(field_s, edge_index, mask, num_iterations):

        field_s[:,mask] = torch.nan
        data = Data(x = field_s.T, edge_index = edge_index)
        transform = T.FeaturePropagation(missing_mask=torch.isnan(data.x), num_iterations = num_iterations)
        data = transform(data)

        field_s_prop = data.x.T.reshape(2,5,66,66,66)
        field_s_prop = field_s_prop[:,:,1:-1,1:-1,1:-1]
        field_s_prop = field_s_prop.cpu().numpy()
        field_s_prop = np.pad(field_s_prop, pad_width=((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)), mode='edge')
        field_s_prop = torch.from_numpy(field_s_prop)
        field_s_prop = field_s_prop.cuda()

        return field_s_prop
    

    def load_tensors_to_gpu(nested_list):

        if isinstance(nested_list, torch.Tensor):
            return nested_list.to('cuda')
        elif isinstance(nested_list, list):
            return [Utils.load_tensors_to_gpu(item) for item in nested_list]
        else:
            return nested_list
        
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
