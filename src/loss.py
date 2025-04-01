import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

import torch
import torch.nn as nn

from src.loss_components import HLLC

class Physical_Loss(nn.Module):

    def __init__(self, gamma, dt, nx, ny, nz, device):

        super(Physical_Loss, self).__init__()

        self.gamma = gamma
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = (1/(self.nx-1))
        self.dy = (1/(self.ny-1))
        self.dz = (1/(self.nz-1))
        self.dt = dt
        self.device = device

    def get_residual(self, output):

        output_m = output.clone()

        output_m[:,4,:,:,:] = output[:,4,:,:,:].clone() + 0.5*(output[:,1,:,:,:].clone()*output[:,1,:,:,:].clone() + output[:,2,:,:,:].clone()*output[:,2,:,:,:].clone() + output[:,3,:,:,:].clone()*output[:,3,:,:,:].clone())

        delta_flux_y, delta_flux_x, delta_flux_z = HLLC.flux_hllc(output_m[0].clone(), self.nx, self.ny, self.nz, self.gamma)

        Q_i = output_m[0].clone()
        U_i = output_m[0].clone()

        Q_f = output_m[1].clone()
        U_f = output_m[1].clone()

        Q_i[1,:,:,:] = Q_i[1,:,:,:].clone() * U_i[0,:,:,:]
        Q_i[2,:,:,:] = Q_i[2,:,:,:].clone() * U_i[0,:,:,:]
        Q_i[3,:,:,:] = Q_i[3,:,:,:].clone() * U_i[0,:,:,:]
        Q_i[4,:,:,:] = Q_i[4,:,:,:].clone() * U_i[0,:,:,:]

        Q_f[1,:,:,:] = Q_f[1,:,:,:].clone() * U_f[0,:,:,:]
        Q_f[2,:,:,:] = Q_f[2,:,:,:].clone() * U_f[0,:,:,:]
        Q_f[3,:,:,:] = Q_f[3,:,:,:].clone() * U_f[0,:,:,:]
        Q_f[4,:,:,:] = Q_f[4,:,:,:].clone() * U_f[0,:,:,:]

        Q_t = (Q_f - Q_i)/self.dt

        flux_x = (1/self.dx)*(delta_flux_x)
        flux_y = (1/self.dy)*(delta_flux_y)
        flux_z = (1/self.dz)*(delta_flux_z)

        residual = Q_t.to(self.device) + flux_x.to(self.device) + flux_y.to(self.device) + flux_z.to(self.device)

        return residual

    def forward(self, x):

        mse_loss = nn.MSELoss()
        residual = Physical_Loss.get_residual(self, x)
        loss = mse_loss(residual, torch.zeros_like(residual))

        return loss


class Data_Loss(nn.Module):
    def __init__(self):
        super(Data_Loss, self).__init__()

    def forward(self, pred, ref, mask):

        masked_pred = torch.masked_select(pred, ~mask)
        masked_ref = torch.masked_select(ref, ~mask)
        loss = nn.functional.mse_loss(masked_pred, masked_ref)

        return loss
