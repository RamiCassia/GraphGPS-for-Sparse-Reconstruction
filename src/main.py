import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

import argparse
import torch
from src.utilities import Utils
from src.training import Train
from src.benchmarks import CNN_BM, GNN_BM, AE_BM
from src.gps import GraphModel
from src.sinusoidal_pe import Sinusoidal_pe

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run_name", type = str)
    parser.add_argument("--seed", type = int)
    parser.add_argument("--con", type = int)
    parser.add_argument("--nepochs", type = int)
    parser.add_argument("--nxyz", type = int)
    parser.add_argument("--sparsity", type = int)
    parser.add_argument("--lr", type = float)
    parser.add_argument("--init", type = str)
    parser.add_argument("--arch", type = str)
    parser.add_argument("--attn", type = str)
    parser.add_argument("--mp", type = str)
    parser.add_argument("--pe", type = str)
    parser.add_argument("--masked_proj", action="store_true")
    parser.add_argument("--uncont_mp", action="store_true")
    parser.add_argument("--connectivity", type = int)
    parser.add_argument("--channels", type = int)
    parser.add_argument("--heads", type = int)
    parser.add_argument("--mamba_dstate", type = int)
    parser.add_argument("--mamba_expand", type = int)
    parser.add_argument("--trans_kernel", type = str)
    parser.add_argument("--sage_aggr", type = str)
    parser.add_argument("--exp_deg", type = int)
    parser.add_argument("--exp_vnodes", type = int)
    parser.add_argument("--pe_kwl", type = int)
    parser.add_argument("--pe_dim", type = int)
    parser.add_argument("--tend", type = float)
    args=parser.parse_args()
    return args

def main():

    inputs = parse_args()
    result_path = base_path + 'results/'
    data_path = base_path + 'data/'

    seed = inputs.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ##############################################################################################

    run_name = inputs.run_name 

    ##############################################################################################

    arch = inputs.arch #'CNN_BM' 'GPS' 'AE_BM' 'GNN_BM'

    ##############################################################################################

    att_type = inputs.attn #'MAMBA' 'TRANSFORMER' 'TRANSFORMERLIN' 'EXPHORMER' 'MAMBA2' 'NONE'
    mp = inputs.mp #'GAT' 'GCN' 'SAGE' 'NONE' 'GATMOD'
    pe_type = inputs.pe #'random_walk' 'euclidean 'laplace' 'sinusoidal'
    masked_projection = inputs.masked_proj
    dense_graph = inputs.uncont_mp

    kernel = inputs.trans_kernel #'exp' #'gaussian' 'exp' 'elup1'
    sage_aggr = inputs.sage_aggr#'max' #'mean' 'lstm' 'max'

    ##############################################################################################

    initialization = inputs.init #'mean' #'zeros' 'feature_prop'
    nx = inputs.nxyz
    ny = inputs.nxyz
    nz = inputs.nxyz
    n_epoch = inputs.nepochs 
    epochs_to_order = 10000000 
    gamma = 1.4
    lr = inputs.lr 
    print_interval = 50
    pe_dim = inputs.pe_dim
    
    heads = 1 

    d_conv = 4
    d_state = inputs.mamba_dstate
    expand = inputs.mamba_expand 

    headdim = inputs.heads

    if arch == 'AE_BM':
        channels = 512
    else:
        channels = inputs.channels

    dt = 0.0005

    data_sparsity = 'random_' + str(inputs.sparsity) + '_con_' + str(inputs.con) + '_nxyz_' + str(inputs.nxyz) + '_dt_' + str(dt) + '_tend_' + str(inputs.tend) + '_p_' + str(inputs.connectivity)

    #############################################################################################

    path_fig, path_model, path_output, path_stats, path_parameters = Utils.create_project(result_path, run_name)
    field = torch.load(data_path + data_sparsity + '/field.pt')
    field_s = torch.load(data_path + data_sparsity + '/field_s.pt')
    mask_data = torch.load(data_path + data_sparsity + '/mask.pt')
    edge_indices_list = torch.load(data_path + data_sparsity + '/edge_list.pt')
    bool_mask_list = torch.load(data_path + data_sparsity + '/bool_list.pt')


    if initialization == 'zeros':
        field_s = Utils.change_filler_value(field_s, bool_mask_list[0], 0)
    elif initialization == 'feature_prop':
        field_s = Utils.feature_propagation(field_s.reshape(10, (nx + 2)*(ny + 2)*(nz + 2)), edge_indices_list[-1], mask_data.reshape((nx + 2)*(ny + 2)*(nz + 2)), num_iterations = 500)
    elif initialization == 'mean':
        pass

    edge_indices_list_with_loops = []
    for i in range(len(edge_indices_list)):
        edge_index = Utils.add_masked_self_loops(field_s, edge_indices_list[i], bool_mask_list[-1])
        edge_indices_list_with_loops.append(edge_index)

    if pe_type == 'random_walk':
        pe = Utils.random_walk_pe(torch.reshape(field_s, (10, (nx + 2)*(ny + 2)*(nz + 2))), edge_indices_list[-1], walk_length = inputs.pe_kwl)
    elif pe_type == 'laplace':
        pe = Utils.laplacePE(torch.reshape(field_s, (10, (nx + 2)*(ny + 2)*(nz + 2))), edge_indices_list[-1], k = inputs.pe_kwl) #k=20
    elif pe_type == 'euclidean':
        pe = Utils.euclid_pe(nx + 2)
    elif pe_type == 'sinusoidal':
        pe = Sinusoidal_pe(5)(field_s, (nx + 2), (ny + 2), nz + 2)


    pe_dim_in = pe.shape[1]
    pe = torch.tensor(pe).to('cuda')
    pe = pe.to(torch.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edge_indices_list_gpu = [tensor.to(device) for tensor in edge_indices_list_with_loops]
    bool_mask_list_gpu = [tensor.view(-1).to(device) for tensor in bool_mask_list]

    if att_type == 'EXPHORMER':

        num_layers = len(edge_indices_list)
        degree = inputs.exp_deg 
        algorithm = 'Random-d'
        num_virtual_nodes = inputs.exp_vnodes
        num_features = channels
        nxyz = nx
        exphormer_chars_list = torch.load(data_path + 'expander_graphs/num_layers_' + str(num_layers) + '_expdegree_' + str(degree) +
                                    '_expalgorithm_' + algorithm + '_numvirtnodes_' + str(num_virtual_nodes) + '_hidden_channels_'
                                    + str(num_features) + '_nxyz_' + str(nxyz) + '.pth')
        exphormer_chars_list = Utils.load_tensors_to_gpu(exphormer_chars_list)
    else:
        exphormer_chars_list = [0]*len(edge_indices_list)


    param_dict = {'configurations':inputs.con, 'seed':seed, 'run_name':run_name, 'arch':arch, 'initialization':initialization, 'learning_rate':lr, 'num_epoch':n_epoch , 'channels':channels, 'gamma':gamma, 'dt':dt, 'nx':nx, 'ny':ny, 'nz':nz, 'connectivity':inputs.connectivity}

    if arch == 'GPS':
        param_dict = param_dict | {'global_attention':att_type, 'mp_type':mp, 'pe_type':pe_type, 'pe_dim':pe_dim, 'pe_dim_in':pe_dim_in, 'masked_projection':masked_projection}
    if mp != 'NONE':
        param_dict = param_dict | {'uncontrolled_mp' : dense_graph, 'heads_mp':heads}
    if mp == 'SAGE':
        param_dict = param_dict | {'sage_aggregation':sage_aggr}
    if att_type != 'NONE':
        param_dict = param_dict | {'global_heads':headdim}
    if att_type == 'MAMBA2' or att_type == 'MAMBA':
        param_dict = param_dict | {'mamba_d_conv':d_conv, 'mamba_d_state':d_state, 'mamba_expand':expand}
    if att_type == 'EXPHORMER':
        param_dict = param_dict | {'exphormer_alg':algorithm, 'exphormer_virt_nodes':num_virtual_nodes, 'exphormer_degree':degree}
    if att_type == 'TRANSFORMERLIN':
        param_dict = param_dict | {'kernel_approximation':kernel}

    Utils.dict_to_txt(path_parameters, param_dict)

    if arch == 'GPS':
        model = GraphModel(channels=channels, pe_dim_in = pe_dim_in, pe_dim=pe_dim, att_type=att_type, d_conv=d_conv, d_state=d_state, heads = heads, expand = expand, headdim = headdim, edge_indices_list = edge_indices_list_gpu, bool_mask_list = bool_mask_list_gpu, shape = (2,5,ny,nx,nz), mp = mp, exphormer_chars_list = exphormer_chars_list, dense_graph = dense_graph, masked_projection = masked_projection, kernel = kernel, sage_aggr = sage_aggr)
    elif arch == 'CNN_BM':
        model = CNN_BM()
        Train.initialize_weights(model, 'xavier_normal')
    elif arch == 'GNN_BM':
        model = GNN_BM(channels = channels, edge_indices_list = edge_indices_list_gpu, shape = (2,5,ny,nx,nz), heads = heads)
    elif arch == 'AE_BM':
        model = AE_BM(channels = channels, num_layers = 4, shape = (2,5,ny,nx,nz))

    Train.train(model, mask_data, field_s, field, n_epoch, gamma, dt, nx, ny, nz, lr, print_interval, arch, pe, epochs_to_order, device, path_fig, path_model, path_output, path_stats)


if __name__ == '__main__':
    main() 
