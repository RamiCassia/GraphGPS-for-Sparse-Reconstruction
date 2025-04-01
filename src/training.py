import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

import torch
import numpy as np
import time
from src.utilities import Utils
from src.plotting import Plot
from src.loss import Physical_Loss, Data_Loss
import torch.nn.functional as F
import torch.optim as optim
import gc

import torch
import torch.nn as nn

class Train():

    def initialize_weights(model, init_type="kaiming_normal"):
      
        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):  
                if init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                elif init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                else:
                    raise ValueError(f"Unknown initialization type: {init_type}")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(init_func) 

    def train(model, mask, field_s, field, n_epoch, gamma, dt, nx, ny, nz, lr, print_interval, arch, pe, epochs_to_order, device, path_fig, path_model, path_output, path_stats, weno):

        num_param = Utils.count_parameters(model)
        
        model = model.to(device)
        mask = mask.to(device)
        field = field.to(device)
        field_s = field_s.to(device)
        Z_values = torch.ones((nx+2)*(ny+2)*(nz+2), 3).to(device)
        pe = pe.to(device)


        if arch == 'CNN_BM':
            x_in = field_s[:, :, 1:-1, 1:-1, 1:-1].clone()
            field_s_data = field_s[:,:,1:-1, 1:-1, 1:-1].clone()
            mask_data = mask[1:-1, 1:-1, 1:-1].clone()
        else:
            t, c, h, w, d = field_s.shape
            x_in = field_s.reshape(t * c, h * w * d).T
            field_s_data = field_s[:,:,1:-1, 1:-1, 1:-1].clone()
            mask_data = mask[1:-1, 1:-1, 1:-1].clone()


        loss_d_list = []
        loss_p_list = []
        error_list = []
        global_conts_list = []
        mem_allocated_list = []
        mem_reserved_list = []
        max_mem_allocated_list = []
        max_mem_reserved_list = []
        time_list = []


        L_p = Physical_Loss(gamma, dt, nx, ny, nz, weno, device)
        L_d = Data_Loss()

        optimizer_p = optim.Adam(model.parameters(), lr=lr)
        optimizer_d = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(n_epoch):

            if (epoch) == 0:
                error = torch.norm((field_s_data - field), p = 2)/torch.norm((field), p = 2)
                error_list.append(error.cpu().detach().numpy())
                Plot.plot_unfolded(field_s_data.cpu().detach().numpy(), field.cpu().detach().numpy(), 'Error = ' + str(error.cpu().detach().numpy()) + ' @ Epoch ' + str(0), save = True, epoch = 0, path = path_fig)

            t_i = time.time()

            if arch == 'GPS':
                x, global_conts, h_att_list, h_local_list = model(x_in, pe, Z_values, epoch, epochs_to_order)
                global_conts_list.append(global_conts)

            else:
                x = model(x_in)

            #########################

            loss_TV = 0
            loss_p = L_p(x)
            optimizer_p.zero_grad()
            loss_p.backward(retain_graph = True)

            loss_d = L_d(x, field_s_data, mask_data)
            optimizer_d.zero_grad()
            loss_d.backward()

            optimizer_p.step()
            optimizer_d.step()

            t_f = time.time()

            time_epoch = t_f - t_i
            time_list.append(np.round(time_epoch,4))

            loss_d_list.append(loss_d.item())
            loss_p_list.append(loss_p.item())

            ###########################

            if arch == 'GPS':
                Z_values = Utils.get_Z(F.pad(x, (1,1,1,1,1,1), mode='replicate'))

            error = torch.norm((x - field), p = 2)/torch.norm((field), p = 2)
            error_list.append(error.cpu().detach().numpy())

            mem_allocated_list.append(torch.cuda.memory_allocated() / 1e9)
            mem_reserved_list.append(torch.cuda.memory_reserved() / 1e9)
            max_mem_allocated_list.append(torch.cuda.max_memory_allocated() / 1e9)
            max_mem_reserved_list.append(torch.cuda.max_memory_reserved() / 1e9)

            gc.collect()
            torch.cuda.empty_cache()

            if (epoch) % (print_interval) == 0 and epoch != 0:
                Plot.plot_unfolded(x.cpu().detach().numpy(), field.cpu().detach().numpy(), 'Error = ' + str(error.cpu().detach().numpy()) + ' @ Epoch ' + str(epoch), save = True, epoch = epoch, path = path_fig)
                np.save(path_output + 'output_epoch_' + str(epoch) + '.npy', x.cpu().detach().numpy())

            if epoch == n_epoch - 1:
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_p_state_dict': optimizer_p.state_dict(), 'optimizer_d_state_dict': optimizer_d.state_dict(),
                            'loss_p': loss_p.item(), 'loss_d' : loss_d.item()}, path_model + "checkpoint.pth")
                np.save(path_stats + 'loss_p.npy', np.array(loss_p_list))
                np.save(path_stats + 'loss_d.npy', np.array(loss_d_list))
                np.save(path_stats + 'error.npy', np.array(error_list))
                np.save(path_output + 'output_final.npy', x.cpu().detach().numpy())
                np.save(path_stats + 'mem_allocated.npy', np.array(mem_allocated_list))
                np.save(path_stats + 'mem_reserved.npy', np.array(mem_reserved_list))
                np.save(path_stats + 'max_mem_allocated.npy', np.array(max_mem_allocated_list))
                np.save(path_stats + 'max_mem_reserved.npy', np.array(max_mem_reserved_list))
                np.save(path_stats + 'time.npy', np.array(time_list))
                np.save(path_stats + 'num_param.npy', num_param)
                if arch == 'GPS':
                    np.save(path_stats + 'global_conts.npy', np.array(global_conts_list))
                    np.save(path_stats + 'h_att_list.npy', np.array(h_att_list))
                    np.save(path_stats + 'h_local_list.npy', np.array(h_local_list))

            print("\r", end = '[%d/%d %d%%] Physical Loss: %.10f Data Loss: %.10f TV Reg: %.10f Error: %.10f' % ((epoch), (n_epoch - 1), ((epoch)/(n_epoch)*100.0), loss_p, loss_d, loss_TV, error), flush=True)
