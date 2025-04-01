import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

import torch
import torch.nn.functional as F

class HLLC():

    def get_U_From_Cons(rho, rho_u, rho_v, rho_w, rho_E):

        return torch.stack((rho, rho_u, rho_v, rho_w, rho_E))

    def get_Cons_From_U(X, nx, ny, nz):

        Xresh = X.permute(1,2,3,0)
        rho = Xresh[:,:,:,0]
        rho_u = Xresh[:,:,:,1]
        rho_v = Xresh[:,:,:,2]
        rho_w = Xresh[:,:,:,3]
        rho_E = Xresh[:,:,:,4]

        return rho,rho_u,rho_v,rho_w,rho_E

    def computeOtherVariables(rho, rho_u, rho_v,rho_w, rho_E, gamma=1.4):

        eps = 0

        u = rho_u.clone()/(rho.clone() + eps)
        v = rho_v.clone()/(rho.clone() + eps)
        w = rho_w.clone()/(rho.clone() + eps)
        E = rho_E.clone()/(rho.clone() + eps)
        p = rho.clone()*(gamma-1)*(E - 0.5*(u.clone()*u.clone() + v.clone()*v.clone() + w.clone()*w.clone()))
        a = torch.sqrt(gamma*p.clone()/(rho.clone() + eps))
        H = 0.5*(u.clone()*u.clone() + v.clone()*v.clone() + w.clone()*w.clone()) + a.clone()*a.clone()/(gamma-1)

        return {'r':rho,'u':u,'v':v, 'w':w, 'E':E, 'a':a, 'p':p, 'H':H}

    def fluxEulerPhysique(W, direction):

        rho = W[0,:]
        rho_u = W[1,:]
        rho_v = W[2,:]
        rho_w = W[3,:]
        rho_E = W[4,:]

        out = HLLC.computeOtherVariables(rho, rho_u, rho_v, rho_w, rho_E)
        u,v,w,p = out['u'], out['v'], out['w'], out['p']


        F = torch.zeros_like(W)
        if direction==0:
            F[0,:] = rho_u.clone()
            F[1,:] = rho_u.clone()*u + p
            F[2,:] = rho_u.clone()*v
            F[3,:] = rho_u.clone()*w
            F[4,:] = (rho_E.clone() + p)*u
        elif direction==1:
            F[0,:] = rho_v.clone()
            F[1,:] = rho_v.clone()*u
            F[2,:] = rho_v.clone()*v + p
            F[3,:] = rho_v.clone()*w
            F[4,:] = (rho_E.clone() + p)*v
        elif direction==2:
            F[0,:] = rho_w.clone()
            F[1,:] = rho_w.clone()*u
            F[2,:] = rho_w.clone()*v
            F[3,:] = rho_w.clone()*w + p
            F[4,:] = (rho_E.clone() + p)*w

        return F

    def HLLCsolver(WL, WR, gamma = 1.4):

        rhoL, rho_uL, rho_vL, rho_wL, rho_EL = WL[0,:], WL[1,:], WL[2,:], WL[3,:], WL[4,:]
        rhoR, rho_uR, rho_vR, rho_wR, rho_ER = WR[0,:], WR[1,:], WR[2,:], WR[3,:], WR[4,:]

        out = HLLC.computeOtherVariables(rhoR, rho_uR, rho_vR, rho_wR, rho_ER)
        uR,vR,wR,pR,ER,HR,aR = out['u'], out['v'], out['w'], out['p'], out['E'], out['H'], out['a']

        out = HLLC.computeOtherVariables(rhoL, rho_uL, rho_vL, rho_wL, rho_EL)
        uL,vL,wL,pL,EL,HL,aL = out['u'], out['v'],out['w'], out['p'], out['E'], out['H'], out['a']

        eps = 0

        utilde = (torch.sqrt(rhoL.clone())*uL.clone() + torch.sqrt(rhoR.clone())*uR.clone())/(torch.sqrt(rhoL.clone()) + torch.sqrt(rhoR.clone()) + eps)
        Htilde = (torch.sqrt(rhoL.clone())*HL.clone() + torch.sqrt(rhoR.clone())*HR.clone())/(torch.sqrt(rhoL.clone()) + torch.sqrt(rhoR.clone()) + eps)
        atilde = torch.sqrt((gamma - 1)*(Htilde.clone() - 0.5*utilde.clone()*utilde.clone()))

        SL = utilde.clone() - atilde.clone()
        SR = utilde.clone() + atilde.clone()

        Sstar = (pR - pL + rhoL.clone()*uL*(SL-uL) - rhoR.clone()*uR*(SR-uR) ) / ((rhoL.clone()*(SL - uL) - rhoR.clone()*(SR - uR) + eps))

        Wstar_L = torch.zeros_like(WL)
        coeff = rhoL.clone()*(SL - uL)/(SL - Sstar + eps)
        Wstar_L[0,:] = coeff
        Wstar_L[1,:] = coeff*Sstar
        Wstar_L[2,:] = coeff*vL
        Wstar_L[3,:] = coeff*wL
        Wstar_L[4,:] = coeff*(EL + (Sstar - uL)*(Sstar + pL/(rhoL.clone()*(SL - uL) + eps)))

        Wstar_R = torch.zeros_like(WL)
        coeff = rhoR.clone()*(SR - uR)/(SR - Sstar + eps)
        Wstar_R[0,:] = coeff
        Wstar_R[1,:] = coeff*Sstar
        Wstar_R[2,:] = coeff*vR
        Wstar_R[3,:] = coeff*wR
        Wstar_R[4,:] = coeff*(ER + (Sstar - uR)*(Sstar + pR/(rhoR.clone()*(SR - uR) + eps)))

        stck_SL = SL.unsqueeze(0).expand(5, -1)
        stck_Sstar = Sstar.unsqueeze(0).expand(5, -1)
        stck_SR = SR.unsqueeze(0).expand(5, -1)

        face_flux1 = torch.where(stck_SL>0, HLLC.fluxEulerPhysique(WL,direction=0), 0)
        face_flux2 = torch.where((stck_SL<=0) & (stck_Sstar>=0), HLLC.fluxEulerPhysique(Wstar_L,direction=0), 0)
        face_flux3 = torch.where((stck_SR>0) & (stck_Sstar<0), HLLC.fluxEulerPhysique(Wstar_R,direction=0), 0)
        face_flux4 = torch.where(stck_SR<=0, HLLC.fluxEulerPhysique(WR,direction=0), 0)

        face_flux = face_flux1 + face_flux2 + face_flux3 + face_flux4

        return face_flux

    def computeFluxes(WL, WR, direction, gamma = 1.4):

        if direction == 1:
            WR[[1,2],:] = WR[[2,1],:]
            WL[[1,2],:] = WL[[2,1],:]
        elif direction == 2:
            WR[[1,3],:] = WR[[3,1],:]
            WL[[1,3],:] = WL[[3,1],:]

        face_flux = HLLC.HLLCsolver(WL,WR)

        if direction==1:
            face_flux[[1,2],:] = face_flux[[2,1],:]
            WR[[1,2],:] = WR[[2,1],:]
            WL[[1,2],:] = WL[[2,1],:]
        elif direction == 2:
            face_flux[[1,3],:] = face_flux[[3,1],:]
            WR[[1,3],:] = WR[[3,1],:]
            WL[[1,3],:] = WL[[3,1],:]

        return face_flux

    def flux_hllc(U, nx, ny, nz, gamma = 1.4):

        U[1,:,:,:] = U[0,:,:,:].clone() * U[1,:,:,:].clone()
        U[2,:,:,:] = U[0,:,:,:].clone() * U[2,:,:,:].clone()
        U[3,:,:,:] = U[0,:,:,:].clone() * U[3,:,:,:].clone()
        U[4,:,:,:] = U[0,:,:,:].clone() * U[4,:,:,:].clone()

        rho, rho_u, rho_v, rho_w, rho_E = HLLC.get_Cons_From_U(U.clone(), nx, ny, nz)

        Wup = torch.zeros((5,(nx)*(ny-1)*(nz)))
        Wup[0,:] = rho[:-1,:,:].reshape((-1,))
        Wup[1,:] = rho_u[:-1,:,:].reshape((-1,))
        Wup[2,:] = rho_v[:-1,:,:].reshape((-1,))
        Wup[3,:] = rho_w[:-1,:,:].reshape((-1,))
        Wup[4,:] = rho_E[:-1,:,:].reshape((-1,))

        Wdown = torch.zeros((5,(nx)*(ny-1)*(nz)))
        Wdown[0,:] = rho[1:,:,:].reshape((-1,))
        Wdown[1,:] = rho_u[1:,:,:].reshape((-1,))
        Wdown[2,:] = rho_v[1:,:,:].reshape((-1,))
        Wdown[3,:] = rho_w[1:,:,:].reshape((-1,))
        Wdown[4,:] = rho_E[1:,:,:].reshape((-1,))

        Wleft = torch.zeros((5,(nx-1)*(ny)*(nz)))
        Wleft[0,:] = rho[:,:-1,:].reshape((-1,))
        Wleft[1,:] = rho_u[:,:-1,:].reshape((-1,))
        Wleft[2,:] = rho_v[:,:-1,:].reshape((-1,))
        Wleft[3,:] = rho_w[:,:-1,:].reshape((-1,))
        Wleft[4,:] = rho_E[:,:-1,:].reshape((-1,))

        Wright = torch.zeros((5,(nx-1)*(ny)*(nz)))
        Wright[0,:] = rho[:,1:,:].reshape((-1,))
        Wright[1,:] = rho_u[:,1:,:].reshape((-1,))
        Wright[2,:] = rho_v[:,1:,:].reshape((-1,))
        Wright[3,:] = rho_w[:,1:,:].reshape((-1,))
        Wright[4,:] = rho_E[:,1:,:].reshape((-1,))

        Wtop = torch.zeros((5,(nx)*(ny)*(nz-1)))
        Wtop[0,:] = rho[:,:,:-1].reshape((-1,))
        Wtop[1,:] = rho_u[:,:,:-1].reshape((-1,))
        Wtop[2,:] = rho_v[:,:,:-1].reshape((-1,))
        Wtop[3,:] = rho_w[:,:,:-1].reshape((-1,))
        Wtop[4,:] = rho_E[:,:,:-1].reshape((-1,))

        Wbot = torch.zeros((5,(nx)*(ny)*(nz-1)))
        Wbot[0,:] = rho[:,:,1:].reshape((-1,))
        Wbot[1,:] = rho_u[:,:,1:].reshape((-1,))
        Wbot[2,:] = rho_v[:,:,1:].reshape((-1,))
        Wbot[3,:] = rho_w[:,:,1:].reshape((-1,))
        Wbot[4,:] = rho_E[:,:,1:].reshape((-1,))

        fluxes_down = torch.zeros((5,ny+1,nx,nz))
        fluxes_down_inner  = HLLC.computeFluxes(Wup,Wdown,direction=1)
        fluxes_down[:,1:-1,:,:] = fluxes_down_inner.reshape((5,ny-1,nx,nz))

        fluxes_right = torch.zeros((5,ny,nx+1,nz))
        fluxes_right_inner = HLLC.computeFluxes(Wleft,Wright,direction=0)
        fluxes_right[:,:,1:-1,:] = fluxes_right_inner.reshape((5,ny,nx-1,nz))

        fluxes_bot = torch.zeros((5,ny,nx,nz+1))
        fluxes_bot_inner = HLLC.computeFluxes(Wtop,Wbot,direction=2)
        fluxes_bot[:,:,:,1:-1] = fluxes_bot_inner.reshape((5,ny,nx,nz-1))

        W = torch.zeros((5,ny,nz))
        W[0,:,:] = rho[:,0,:]
        W[1,:,:] = rho_u[:,0,:]
        W[2,:,:] = rho_v[:,0,:]
        W[3,:,:] = rho_w[:,0,:]
        W[4,:,:] = rho_E[:,0,:]
        fluxes_right_outer = HLLC.fluxEulerPhysique(W, direction=0)
        fluxes_right[:,:,0,:] = fluxes_right_outer.reshape((5,ny,nz))

        W = torch.zeros((5,ny,nz))
        W[0,:,:] = rho[:,-1,:]
        W[1,:,:] = rho_u[:,-1,:]
        W[2,:,:] = rho_v[:,-1,:]
        W[3,:,:] = rho_w[:,-1,:]
        W[4,:,:] = rho_E[:,-1,:]
        fluxes_right_outer = HLLC.fluxEulerPhysique(W, direction=0)
        fluxes_right[:,:,-1,:] = fluxes_right_outer.reshape((5,ny,nz))

        W = torch.zeros((5,nx,nz))
        W[0,:,:] = rho[0,:,:]
        W[1,:,:] = rho_u[0,:,:]
        W[2,:,:] = rho_v[0,:,:]
        W[3,:,:] = rho_w[0,:,:]
        W[4,:,:] = rho_E[0,:,:]
        fluxes_down_outer  = HLLC.fluxEulerPhysique(W, direction=1)
        fluxes_down[:,0,:,:] = fluxes_down_outer.reshape((5,nx,nz))

        W = torch.zeros((5,nx,nz))
        W[0,:,:] = rho[-1,:,:]
        W[1,:,:] = rho_u[-1,:,:]
        W[2,:,:] = rho_v[-1,:,:]
        W[3,:,:] = rho_w[-1,:,:]
        W[4,:,:] = rho_E[-1,:,:]
        fluxes_down_outer  = HLLC.fluxEulerPhysique(W, direction=1)
        fluxes_down[:,-1,:,:] = fluxes_down_outer.reshape((5,nx,nz))

        W = torch.zeros((5,nx,ny))
        W[0,:,:] = rho[:,:,0]
        W[1,:,:] = rho_u[:,:,0]
        W[2,:,:] = rho_v[:,:,0]
        W[3,:,:] = rho_w[:,:,0]
        W[4,:,:] = rho_E[:,:,0]
        fluxes_bot_outer = HLLC.fluxEulerPhysique(W, direction=2)
        fluxes_bot[:,:,:,0] = fluxes_bot_outer.reshape((5,nx,ny))

        W = torch.zeros((5,nx,ny))
        W[0,:,:] = rho[:,:,-1]
        W[1,:,:] = rho_u[:,:,-1]
        W[2,:,:] = rho_v[:,:,-1]
        W[3,:,:] = rho_w[:,:,-1]
        W[4,:,:] = rho_E[:,:,-1]
        fluxes_bot_outer = HLLC.fluxEulerPhysique(W, direction=2)
        fluxes_bot[:,:,:,-1] = fluxes_bot_outer.reshape((5,nx,ny))

        flux_diff_y, flux_diff_x, flux_diff_z = (fluxes_down[:,1:,:,:] - fluxes_down[:,:-1,:,:]), (fluxes_right[:,:,1:,:] - fluxes_right[:,:,:-1,:]), (fluxes_bot[:,:,:,1:] - fluxes_bot[:,:,:,:-1])

        return flux_diff_y, flux_diff_x, flux_diff_z
