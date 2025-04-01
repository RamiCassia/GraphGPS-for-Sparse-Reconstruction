import cupy as np
import matplotlib.pyplot as plt

gamma= 1.4

def weno5_reconstruction(arr, axis):
  
    epsilon = 1e-6 
    C = np.array([1/10, 6/10, 3/10])

    padded = np.pad(arr, [(2, 2) if i == axis else (0, 0) for i in range(3)], mode='reflect')
    stencils = [np.roll(padded, -shift, axis=axis) for shift in range(-2, 3)]

    beta = [(13/12) * (stencils[0] - 2 * stencils[1] + stencils[2])**2 + (1/4) * (stencils[0] - 4*stencils[1] + 3*stencils[2])**2,
            (13/12) * (stencils[1] - 2 * stencils[2] + stencils[3])**2 + (1/4) * (stencils[1] - stencils[3])**2,
            (13/12) * (stencils[2] - 2 * stencils[3] + stencils[4])**2 + (1/4) * (3 * stencils[2] - 4 * stencils[3] + stencils[4])**2]

    beta = np.array(beta) 
    alpha = C[:, None, None, None] / (epsilon + beta)**2 
    alpha_sum = np.sum(alpha, axis=0) 
    omega = alpha / alpha_sum  

    out_L = (
        omega[0] * (2/6 * stencils[0] - 7/6 * stencils[1] + 11/6 * stencils[2]) +
        omega[1] * (-1/6 * stencils[1] + 5/6 * stencils[2] + 2/6 * stencils[3]) +
        omega[2] * (2/6 * stencils[2] + 5/6 * stencils[3] - 1/6 * stencils[4])
    )
    out_R = (
        omega[0] * (-1/6 * stencils[0] + 5/6 * stencils[1] + 2/6 * stencils[2]) +
        omega[1] * (2/6 * stencils[1] + 5/6 * stencils[2] - 1/6 * stencils[3]) +
        omega[2] * (11/6 * stencils[2] - 7/6 * stencils[3] + 2/6 * stencils[4])
    )

    slicesL = [slice(2, -3) if i == axis else slice(None) for i in range(3)]
    slicesR = [slice(3, -2) if i == axis else slice(None) for i in range(3)]
    out_L, out_R = out_L[tuple(slicesL)], out_R[tuple(slicesR)]

    return out_L, out_R


def computeOtherVariables(rho, rho_u, rho_v,rho_w, rho_E):
    u = rho_u/rho
    v = rho_v/rho
    w = rho_w/rho
    E = rho_E/rho
    p = rho*(gamma-1)*(E - 0.5*(u*u + v*v + w*w))
    a = np.sqrt(gamma*p/rho)
    H = 0.5*(u*u + v*v + w*w) + a*a/(gamma-1)

    return {'r':rho,'u':u,'v':v, 'w':w, 'E':E, 'a':a, 'p':p, 'H':H}

def fluxEulerPhysique(W, direction):
    rho = W[0,:]
    rho_u = W[1,:]
    rho_v = W[2,:]
    rho_w = W[3,:]
    rho_E = W[4,:]

    out = computeOtherVariables(rho, rho_u, rho_v, rho_w, rho_E)
    u,v,w,p = out['u'], out['v'], out['w'], out['p']

    F = np.zeros_like(W)
    if direction==0:
      F[0,:] = rho_u
      F[1,:] = rho_u*u + p
      F[2,:] = rho_u*v
      F[3,:] = rho_u*w
      F[4,:] = (rho_E + p)*u
    elif direction==1:
      F[0,:] = rho_v
      F[1,:] = rho_v*u
      F[2,:] = rho_v*v + p
      F[3,:] = rho_v*w
      F[4,:] = (rho_E + p)*v
    elif direction==2:
      F[0,:] = rho_w
      F[1,:] = rho_w*u
      F[2,:] = rho_w*v
      F[3,:] = rho_w*w + p
      F[4,:] = (rho_E + p)*w

    return F

def HLLCsolver(WL,WR):

    rhoL, rho_uL, rho_vL, rho_wL, rho_EL = WL[0,:], WL[1,:], WL[2,:], WL[3,:], WL[4,:]
    rhoR, rho_uR, rho_vR, rho_wR, rho_ER = WR[0,:], WR[1,:], WR[2,:], WR[3,:], WR[4,:]
    out = computeOtherVariables(rhoR, rho_uR, rho_vR, rho_wR, rho_ER)
    uR,vR,wR,pR,ER,HR,aR = out['u'], out['v'], out['w'], out['p'], out['E'], out['H'], out['a']
    out = computeOtherVariables(rhoL, rho_uL, rho_vL, rho_wL, rho_EL)
    uL,vL,wL,pL,EL,HL,aL = out['u'], out['v'],out['w'], out['p'], out['E'], out['H'], out['a']

    face_flux = np.zeros_like(WL)

    utilde = (np.sqrt(rhoL)*uL + np.sqrt(rhoR)*uR)/(np.sqrt(rhoL) + np.sqrt(rhoR))
    Htilde = (np.sqrt(rhoL)*HL + np.sqrt(rhoR)*HR)/(np.sqrt(rhoL) + np.sqrt(rhoR))
    atilde = np.sqrt( (gamma-1)*(Htilde-0.5*utilde*utilde) )
    SL = utilde - atilde
    SR = utilde + atilde

    Sstar = ( pR-pL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR) ) / ( rhoL*(SL-uL) - rhoR*(SR-uR) )

    Wstar_L = np.zeros_like(WL)
    coeff = rhoL*(SL-uL)/(SL - Sstar)
    Wstar_L[0,:] = coeff
    Wstar_L[1,:] = coeff*Sstar
    Wstar_L[2,:] = coeff*vL
    Wstar_L[3,:] = coeff*wL
    Wstar_L[4,:] = coeff*( EL+ (Sstar-uL)*(Sstar + pL/(rhoL*(SL-uL))) )

    Wstar_R = np.zeros_like(WL)
    coeff = rhoR*(SR-uR)/(SR - Sstar)
    Wstar_R[0,:] = coeff
    Wstar_R[1,:] = coeff*Sstar
    Wstar_R[2,:] = coeff*vR
    Wstar_R[3,:] = coeff*wR
    Wstar_R[4,:] = coeff*( ER+ (Sstar-uR)*(Sstar + pR/(rhoR*(SR-uR))) )

    total=0

    I=np.where(SL>0,1,0)
    face_flux1 = np.where(SL>0, fluxEulerPhysique(WL,direction=0), 0)
    total = total + np.sum(I)

    I=np.where((SL<=0) & (Sstar>=0),1,0)
    face_flux2 = np.where((SL<=0) & (Sstar>=0), fluxEulerPhysique(Wstar_L,direction=0), 0)
    total = total + np.sum(I)

    I=np.where((SR>0) & (Sstar<0),1,0)
    face_flux3 = np.where((SR>0) & (Sstar<0), fluxEulerPhysique(Wstar_R,direction=0), 0)
    total = total + np.sum(I)

    I=np.where(SR<=0,1,0)
    face_flux4 = np.where(SR<=0, fluxEulerPhysique(WR,direction=0), 0)
    total = total + np.sum(I)

    face_flux = face_flux1 + face_flux2 + face_flux3 + face_flux4

    if total != SR.size:
        raise Exception('problem HLL UNRESOLVED CASE')

    return face_flux

def computeFluxes(WL,WR,direction):
    if direction == 1: 
      WR[[1,2],:] = WR[[2,1],:]
      WL[[1,2],:] = WL[[2,1],:]
    elif direction == 2:
      WR[[1,3],:] = WR[[3,1],:]
      WL[[1,3],:] = WL[[3,1],:]

    face_flux = HLLCsolver(WL,WR)

    if direction==1: 
      face_flux[[1,2],:] = face_flux[[2,1],:]
      WR[[1,2],:] = WR[[2,1],:]
      WL[[1,2],:] = WL[[2,1],:]
    elif direction == 2:
      face_flux[[1,3],:] = face_flux[[3,1],:]
      WR[[1,3],:] = WR[[3,1],:]
      WL[[1,3],:] = WL[[3,1],:]

    return face_flux

def modelfun(U, nx, ny, nz):

    rho, rho_u, rho_v, rho_w, rho_E = get_Cons_From_U(U, nx, ny, nz)

    rho_L_y, rho_R_y = weno5_reconstruction(rho, axis = 0)
    rho_u_L_y, rho_u_R_y = weno5_reconstruction(rho_u, axis = 0)
    rho_v_L_y, rho_v_R_y = weno5_reconstruction(rho_v, axis = 0)
    rho_w_L_y, rho_w_R_y = weno5_reconstruction(rho_w, axis = 0)
    rho_E_L_y, rho_E_R_y = weno5_reconstruction(rho_E, axis = 0)

    rho_L_x, rho_R_x = weno5_reconstruction(rho, axis = 1)
    rho_u_L_x, rho_u_R_x = weno5_reconstruction(rho_u, axis = 1)
    rho_v_L_x, rho_v_R_x = weno5_reconstruction(rho_v, axis = 1)
    rho_w_L_x, rho_w_R_x = weno5_reconstruction(rho_w, axis = 1)
    rho_E_L_x, rho_E_R_x = weno5_reconstruction(rho_E, axis = 1)

    rho_L_z, rho_R_z = weno5_reconstruction(rho, axis = 2)
    rho_u_L_z, rho_u_R_z = weno5_reconstruction(rho_u, axis = 2)
    rho_v_L_z, rho_v_R_z = weno5_reconstruction(rho_v, axis = 2)
    rho_w_L_z, rho_w_R_z = weno5_reconstruction(rho_w, axis = 2)
    rho_E_L_z, rho_E_R_z = weno5_reconstruction(rho_E, axis = 2)


    Wup = np.zeros((5,(nx)*(ny-1)*(nz)))
    Wup[0,:] = rho_L_y[:,:,:].reshape((-1,))
    Wup[1,:] = rho_u_L_y[:,:,:].reshape((-1,))
    Wup[2,:] = rho_v_L_y[:,:,:].reshape((-1,))
    Wup[3,:] = rho_w_L_y[:,:,:].reshape((-1,))
    Wup[4,:] = rho_E_L_y[:,:,:].reshape((-1,))

    Wdown = np.zeros((5,(nx)*(ny-1)*(nz)))
    Wdown[0,:] = rho_R_y[:,:,:].reshape((-1,))
    Wdown[1,:] = rho_u_R_y[:,:,:].reshape((-1,))
    Wdown[2,:] = rho_v_R_y[:,:,:].reshape((-1,))
    Wdown[3,:] = rho_w_R_y[:,:,:].reshape((-1,))
    Wdown[4,:] = rho_E_R_y[:,:,:].reshape((-1,))

    Wleft = np.zeros((5,(nx-1)*(ny)*(nz)))
    Wleft[0,:] = rho_L_x[:,:,:].reshape((-1,))
    Wleft[1,:] = rho_u_L_x[:,:,:].reshape((-1,))
    Wleft[2,:] = rho_v_L_x[:,:,:].reshape((-1,))
    Wleft[3,:] = rho_w_L_x[:,:,:].reshape((-1,))
    Wleft[4,:] = rho_E_L_x[:,:,:].reshape((-1,))

    Wright = np.zeros((5,(nx-1)*(ny)*(nz)))
    Wright[0,:] = rho_R_x[:,:,:].reshape((-1,))
    Wright[1,:] = rho_u_R_x[:,:,:].reshape((-1,))
    Wright[2,:] = rho_v_R_x[:,:,:].reshape((-1,))
    Wright[3,:] = rho_w_R_x[:,:,:].reshape((-1,))
    Wright[4,:] = rho_E_R_x[:,:,:].reshape((-1,))

    Wtop = np.zeros((5,(nx)*(ny)*(nz-1)))
    Wtop[0,:] = rho_L_z[:,:,:].reshape((-1,))
    Wtop[1,:] = rho_u_L_z[:,:,:].reshape((-1,))
    Wtop[2,:] = rho_v_L_z[:,:,:].reshape((-1,))
    Wtop[3,:] = rho_w_L_z[:,:,:].reshape((-1,))
    Wtop[4,:] = rho_E_L_z[:,:,:].reshape((-1,))

    Wbot = np.zeros((5,(nx)*(ny)*(nz-1)))
    Wbot[0,:] = rho_R_z[:,:,:].reshape((-1,))
    Wbot[1,:] = rho_u_R_z[:,:,:].reshape((-1,))
    Wbot[2,:] = rho_v_R_z[:,:,:].reshape((-1,))
    Wbot[3,:] = rho_w_R_z[:,:,:].reshape((-1,))
    Wbot[4,:] = rho_E_R_z[:,:,:].reshape((-1,))

    fluxes_down = np.zeros((5,ny+1,nx,nz))
    fluxes_down_inner  = computeFluxes(Wup,Wdown,direction=1)
    fluxes_down[:,1:-1,:,:] = fluxes_down_inner.reshape((5,ny-1,nx,nz))

    fluxes_right = np.zeros((5,ny,nx+1,nz))
    fluxes_right_inner = computeFluxes(Wleft,Wright,direction=0)
    fluxes_right[:,:,1:-1,:] = fluxes_right_inner.reshape((5,ny,nx-1,nz))

    fluxes_bot = np.zeros((5,ny,nx,nz+1))
    fluxes_bot_inner = computeFluxes(Wtop,Wbot,direction=2)
    fluxes_bot[:,:,:,1:-1] = fluxes_bot_inner.reshape((5,ny,nx,nz-1))

    W = np.zeros((5,nx,nz))
    W[0,:,:] = rho[0,:,:]
    W[1,:,:] = rho_u[0,:,:]
    W[2,:,:] = rho_v[0,:,:]
    W[3,:,:] = rho_w[0,:,:]
    W[4,:,:] = rho_E[0,:,:]
    fluxes_down_outer  = fluxEulerPhysique(W, direction=1)
    fluxes_down[:,0,:,:] = fluxes_down_outer.reshape((5,nx,nz))

    W = np.zeros((5,nx,nz))
    W[0,:,:] = rho[-1,:,:]
    W[1,:,:] = rho_u[-1,:,:]
    W[2,:,:] = rho_v[-1,:,:]
    W[3,:,:] = rho_w[-1,:,:]
    W[4,:,:] = rho_E[-1,:,:]
    fluxes_down_outer  = fluxEulerPhysique(W, direction=1)
    fluxes_down[:,-1,:,:] = fluxes_down_outer.reshape((5,nx,nz))

    W = np.zeros((5,ny,nz))
    W[0,:,:] = rho[:,0,:]
    W[1,:,:] = rho_u[:,0,:]
    W[2,:,:] = rho_v[:,0,:]
    W[3,:,:] = rho_w[:,0,:]
    W[4,:,:] = rho_E[:,0,:]
    fluxes_right_outer = fluxEulerPhysique(W, direction=0)
    fluxes_right[:,:,0,:] = fluxes_right_outer.reshape((5,ny,nz))

    W = np.zeros((5,ny,nz))
    W[0,:,:] = rho[:,-1,:]
    W[1,:,:] = rho_u[:,-1,:]
    W[2,:,:] = rho_v[:,-1,:]
    W[3,:,:] = rho_w[:,-1,:]
    W[4,:,:] = rho_E[:,-1,:]
    fluxes_right_outer = fluxEulerPhysique(W, direction=0)
    fluxes_right[:,:,-1,:] = fluxes_right_outer.reshape((5,ny,nz))

    W = np.zeros((5,nx,ny))
    W[0,:,:] = rho[:,:,0]
    W[1,:,:] = rho_u[:,:,0]
    W[2,:,:] = rho_v[:,:,0]
    W[3,:,:] = rho_w[:,:,0]
    W[4,:,:] = rho_E[:,:,0]
    fluxes_bot_outer = fluxEulerPhysique(W, direction=2)
    fluxes_bot[:,:,:,0] = fluxes_bot_outer.reshape((5,nx,ny))

    W = np.zeros((5,nx,ny))
    W[0,:,:] = rho[:,:,-1]
    W[1,:,:] = rho_u[:,:,-1]
    W[2,:,:] = rho_v[:,:,-1]
    W[3,:,:] = rho_w[:,:,-1]
    W[4,:,:] = rho_E[:,:,-1]
    fluxes_bot_outer = fluxEulerPhysique(W, direction=2)
    fluxes_bot[:,:,:,-1] = fluxes_bot_outer.reshape((5,nx,ny))

    flux_diff_y, flux_diff_x, flux_diff_z = (fluxes_down[:,1:,:,:] - fluxes_down[:,:-1,:,:]), (fluxes_right[:,:,1:,:] - fluxes_right[:,:,:-1,:]), (fluxes_bot[:,:,:,1:] - fluxes_bot[:,:,:,:-1])

    return flux_diff_y, flux_diff_x, flux_diff_z


def get_U_From_Cons(rho, rho_u, rho_v, rho_w, rho_E):

    return np.stack((rho, rho_u, rho_v, rho_w, rho_E))


def get_Cons_From_U(X, nx, ny, nz):

    Xresh = np.swapaxes(X,0,3)
    Xresh = np.swapaxes(Xresh,0,2)
    Xresh = np.swapaxes(Xresh,0,1)
    rho = Xresh[:,:,:,0]
    rho_u = Xresh[:,:,:,1]
    rho_v = Xresh[:,:,:,2]
    rho_w = Xresh[:,:,:,3]
    rho_E = Xresh[:,:,:,4]
    return rho,rho_u,rho_v,rho_w,rho_E

def get_Cons_From_U_vectorized(X,nx,ny,nz):

    Xresh = X.reshape((ny,nx,nz,4,-1))
    rho = Xresh[:,:,:,0,:]
    rho_u = Xresh[:,:,:,1,:]
    rho_v = Xresh[:,:,:,2,:]
    rho_w = Xresh[:,:,:,3,:]
    rho_E = Xresh[:,:,:,4,:]
    return rho,rho_u,rho_v,rho_w,rho_E

def initialize(min, max, n, con):

    xmin,xmax = min,max
    ymin,ymax = min,max
    zmin,zmax = min,max

    nx,ny,nz = n,n,n

    r0 = np.zeros((ny, nx, nz))
    u0 = np.zeros((ny, nx, nz))
    v0 = np.zeros((ny, nx, nz))
    w0 = np.zeros((ny, nx, nz))
    p0 = np.zeros((ny, nx, nz))
    halfcells = int(nx/2)

    if con == 1: #OSCILLATORY #5 shocks, 6 contacts, 1 rarefaction
        #octant 1: y > 0.5, x > 0.5, z > 0.5
        r0[halfcells:, halfcells:, halfcells:] = 0.622341;
        u0[halfcells:, halfcells:, halfcells:] = 0.1;
        v0[halfcells:, halfcells:, halfcells:] = -0.625862;
        w0[halfcells:, halfcells:, halfcells:] = -0.1;
        p0[halfcells:, halfcells:, halfcells:] = 0.4;

        #octant 2: y > 0.5, x < 0.5, z > 0.5
        r0[halfcells:, :halfcells, halfcells:] = 0.53125;
        u0[halfcells:, :halfcells, halfcells:] = 0.1;
        v0[halfcells:, :halfcells, halfcells:] = 0.1;
        w0[halfcells:, :halfcells, halfcells:] = -0.827607;
        p0[halfcells:, :halfcells, halfcells:] = 0.4;

        #octant 3: y < 0.5, x < 0.5, z > 0.5
        r0[:halfcells, :halfcells, halfcells:] = 0.622341;
        u0[:halfcells, :halfcells, halfcells:] = 0.825862;
        v0[:halfcells, :halfcells, halfcells:] = 0.1;
        w0[:halfcells, :halfcells, halfcells:] = -0.1;
        p0[:halfcells, :halfcells, halfcells:] = 0.4;

        #octant 4: y < 0.5, x > 0.5, z > 0.5
        r0[:halfcells, halfcells:, halfcells:] = 1.221896;
        u0[:halfcells, halfcells:, halfcells:] = 0.1;
        v0[:halfcells, halfcells:, halfcells:] = 0.1;
        w0[:halfcells, halfcells:, halfcells:] = -0.1;
        p0[:halfcells, halfcells:, halfcells:] = 1.068254;

        #octant 5: y > 0.5, x > 0.5, z < 0.5
        r0[halfcells:, halfcells:, :halfcells] = 0.519705;
        u0[halfcells:, halfcells:, :halfcells] = 0.825864;
        v0[halfcells:, halfcells:, :halfcells] = 0.1;
        w0[halfcells:, halfcells:, :halfcells] = -0.1;
        p0[halfcells:, halfcells:, :halfcells] = 0.4;

        #octant 6: y > 0.5, x < 0.5, z < 0.5
        r0[halfcells:, :halfcells, :halfcells] = 1.0;
        u0[halfcells:, :halfcells, :halfcells] = 0.1;
        v0[halfcells:, :halfcells, :halfcells] = 0.1;
        w0[halfcells:, :halfcells, :halfcells] = -0.1;
        p0[halfcells:, :halfcells, :halfcells] = 1.0;

        #octant 7: y < 0.5, x < 0.5, z < 0.5
        r0[:halfcells, :halfcells, :halfcells] = 0.53125;
        u0[:halfcells, :halfcells, :halfcells] = 0.1;
        v0[:halfcells, :halfcells, :halfcells] = 0.827607;
        w0[:halfcells, :halfcells, :halfcells] = -0.1;
        p0[:halfcells, :halfcells, :halfcells] = 0.4;

        #octant 8: y < 0.5, x > 0.5, z < 0.5
        r0[:halfcells, halfcells:, :halfcells] = 0.622341;
        u0[:halfcells, halfcells:, :halfcells] = 0.1;
        v0[:halfcells, halfcells:, :halfcells] = 0.1;
        w0[:halfcells, halfcells:, :halfcells] = -0.625862;
        p0[:halfcells, halfcells:, :halfcells] = 0.4;


    if con == 2: #LOW TIMESTEP REQUIRED #12 shocks
        #octant 1: y > 0.5, x > 0.5, z > 0.5
        r0[halfcells:, halfcells:, halfcells:] = 0.506475;
        u0[halfcells:, halfcells:, halfcells:] = 0.0;
        v0[halfcells:, halfcells:, halfcells:] = 0.0;
        w0[halfcells:, halfcells:, halfcells:] = 0.0;
        p0[halfcells:, halfcells:, halfcells:] = 0.35;

        #octant 2: y > 0.5, x < 0.5, z > 0.5
        r0[halfcells:, :halfcells, halfcells:] = 1.1;
        u0[halfcells:, :halfcells, halfcells:] = 0.893871;
        v0[halfcells:, :halfcells, halfcells:] = 0.0;
        w0[halfcells:, :halfcells, halfcells:] = 0.0;
        p0[halfcells:, :halfcells, halfcells:] = 1.1;

        #octant 3: y < 0.5, x < 0.5, z > 0.5
        r0[:halfcells, :halfcells, halfcells:] = 0.506475;
        u0[:halfcells, :halfcells, halfcells:] = 0.893871;
        v0[:halfcells, :halfcells, halfcells:] = 0.893871;
        w0[:halfcells, :halfcells, halfcells:] = 0.0;
        p0[:halfcells, :halfcells, halfcells:] = 0.35;

        #octant 4: y < 0.5, x > 0.5, z > 0.5
        r0[:halfcells, halfcells:, halfcells:] = 1.1;
        u0[:halfcells, halfcells:, halfcells:] = 0.0;
        v0[:halfcells, halfcells:, halfcells:] = 0.893871;
        w0[:halfcells, halfcells:, halfcells:] = 0.0;
        p0[:halfcells, halfcells:, halfcells:] = 1.1;

        #octant 5: y > 0.5, x > 0.5, z < 0.5
        r0[halfcells:, halfcells:, :halfcells] = 1.1;
        u0[halfcells:, halfcells:, :halfcells] = 0.0;
        v0[halfcells:, halfcells:, :halfcells] = 0.0;
        w0[halfcells:, halfcells:, :halfcells] = 0.893871;
        p0[halfcells:, halfcells:, :halfcells] = 1.1;

        #octant 6: y > 0.5, x < 0.5, z < 0.5
        r0[halfcells:, :halfcells, :halfcells] = 0.506475;
        u0[halfcells:, :halfcells, :halfcells] = 0.893871;
        v0[halfcells:, :halfcells, :halfcells] = 0.0;
        w0[halfcells:, :halfcells, :halfcells] = 0.893871;
        p0[halfcells:, :halfcells, :halfcells] = 0.35;

        #octant 7: y < 0.5, x < 0.5, z < 0.5
        r0[:halfcells, :halfcells, :halfcells] = 1.1;
        u0[:halfcells, :halfcells, :halfcells] = 0.893871;
        v0[:halfcells, :halfcells, :halfcells] = 0.893871;
        w0[:halfcells, :halfcells, :halfcells] = 0.893871;
        p0[:halfcells, :halfcells, :halfcells] = 1.1;

        #octant 8: y < 0.5, x > 0.5, z < 0.5
        r0[:halfcells, halfcells:, :halfcells] = 0.506475;
        u0[:halfcells, halfcells:, :halfcells] = 0.0;
        v0[:halfcells, halfcells:, :halfcells] = 0.893871;
        w0[:halfcells, halfcells:, :halfcells] = 0.893871;
        p0[:halfcells, halfcells:, :halfcells] = 0.35;


    if con == 3: #4 shocks 8 rarefacttions
        #octant 1: y > 0.5, x > 0.5, z > 0.5
        r0[halfcells:, halfcells:, halfcells:] = 1.0;
        u0[halfcells:, halfcells:, halfcells:] = 0.0;
        v0[halfcells:, halfcells:, halfcells:] = 0.0;
        w0[halfcells:, halfcells:, halfcells:] = -0.788080;
        p0[halfcells:, halfcells:, halfcells:] = 1.0;

        #octant 2: y > 0.5, x < 0.5, z > 0.5
        r0[halfcells:, :halfcells, halfcells:] = 0.5;
        u0[halfcells:, :halfcells, halfcells:] = -0.765833;
        v0[halfcells:, :halfcells, halfcells:] = 0.0;
        w0[halfcells:, :halfcells, halfcells:] = -0.788080;
        p0[halfcells:, :halfcells, halfcells:] = 0.378929;

        #octant 3: y < 0.5, x < 0.5, z > 0.5
        r0[:halfcells, :halfcells, halfcells:] = 1.0;
        u0[:halfcells, :halfcells, halfcells:] = -0.765833;
        v0[:halfcells, :halfcells, halfcells:] = -0.765833;
        w0[:halfcells, :halfcells, halfcells:] = -0.788080;
        p0[:halfcells, :halfcells, halfcells:] = 1.0;

        #octant 4: y < 0.5, x > 0.5, z > 0.5
        r0[:halfcells, halfcells:, halfcells:] = 0.5;
        u0[:halfcells, halfcells:, halfcells:] = 0.0;
        v0[:halfcells, halfcells:, halfcells:] = -0.765833;
        w0[:halfcells, halfcells:, halfcells:] = -0.788080;
        p0[:halfcells, halfcells:, halfcells:] = 0.378929;

        #octant 5: y > 0.5, x > 0.5, z < 0.5
        r0[halfcells:, halfcells:, :halfcells] = 0.5;
        u0[halfcells:, halfcells:, :halfcells] = 0.0;
        v0[halfcells:, halfcells:, :halfcells] = 0.0;
        w0[halfcells:, halfcells:, :halfcells] = 0.0;
        p0[halfcells:, halfcells:, :halfcells] = 0.378929;

        #octant 6: y > 0.5, x < 0.5, z < 0.5
        r0[halfcells:, :halfcells, :halfcells] = 1.0;
        u0[halfcells:, :halfcells, :halfcells] = -0.765833;
        v0[halfcells:, :halfcells, :halfcells] = 0.0;
        w0[halfcells:, :halfcells, :halfcells] = 0.0;
        p0[halfcells:, :halfcells, :halfcells] = 1.0;

        #octant 7: y < 0.5, x < 0.5, z < 0.5
        r0[:halfcells, :halfcells, :halfcells] = 0.5;
        u0[:halfcells, :halfcells, :halfcells] = -0.765833;
        v0[:halfcells, :halfcells, :halfcells] = -0.765833;
        w0[:halfcells, :halfcells, :halfcells] = 0.0;
        p0[:halfcells, :halfcells, :halfcells] = 0.378929;

        #octant 8: y < 0.5, x > 0.5, z < 0.5
        r0[:halfcells, halfcells:, :halfcells] = 1.0;
        u0[:halfcells, halfcells:, :halfcells] = 0.0;
        v0[:halfcells, halfcells:, :halfcells] = -0.765833;
        w0[:halfcells, halfcells:, :halfcells] = 0.0;
        p0[:halfcells, halfcells:, :halfcells] = 1.0;

    if con == 4: #2 shocks, 8 contacts, 2 rarefactions
        #octant 1: y > 0.5, x > 0.5, z > 0.5
        r0[halfcells:, halfcells:, halfcells:] = 1.5;
        u0[halfcells:, halfcells:, halfcells:] = 0.1;
        v0[halfcells:, halfcells:, halfcells:] = -0.15;
        w0[halfcells:, halfcells:, halfcells:] = -0.2;
        p0[halfcells:, halfcells:, halfcells:] = 1.0;

        #octant 2: y > 0.5, x < 0.5, z > 0.5
        r0[halfcells:, :halfcells, halfcells:] = 1.0;
        u0[halfcells:, :halfcells, halfcells:] = 0.1;
        v0[halfcells:, :halfcells, halfcells:] = 0.15;
        w0[halfcells:, :halfcells, halfcells:] = -0.3;
        p0[halfcells:, :halfcells, halfcells:] = 1.0;

        #octant 3: y < 0.5, x < 0.5, z > 0.5
        r0[:halfcells, :halfcells, halfcells:] = 1.5;
        u0[:halfcells, :halfcells, halfcells:] = -0.1;
        v0[:halfcells, :halfcells, halfcells:] = 0.15;
        w0[:halfcells, :halfcells, halfcells:] = -0.5;
        p0[:halfcells, :halfcells, halfcells:] = 1.0;

        #octant 4: y < 0.5, x > 0.5, z > 0.5
        r0[:halfcells, halfcells:, halfcells:] = 1.0;
        u0[:halfcells, halfcells:, halfcells:] = -0.1;
        v0[:halfcells, halfcells:, halfcells:] = -0.15;
        w0[:halfcells, halfcells:, halfcells:] = 1.2;
        p0[:halfcells, halfcells:, halfcells:] = 1.0;

        #octant 5: y > 0.5, x > 0.5, z < 0.5
        r0[halfcells:, halfcells:, :halfcells] = 0.796875;
        u0[halfcells:, halfcells:, :halfcells] = 0.1;
        v0[halfcells:, halfcells:, :halfcells] = -0.15;
        w0[halfcells:, halfcells:, :halfcells] = 0.394089;
        p0[halfcells:, halfcells:, :halfcells] = 0.4;

        #octant 6: y > 0.5, x < 0.5, z < 0.5
        r0[halfcells:, :halfcells, :halfcells] = 0.519705;
        u0[halfcells:, :halfcells, :halfcells] = 0.1;
        v0[halfcells:, :halfcells, :halfcells] = 0.15;
        w0[halfcells:, :halfcells, :halfcells] = -1.025864;
        p0[halfcells:, :halfcells, :halfcells] = 0.4;

        #octant 7: y < 0.5, x < 0.5, z < 0.5
        r0[:halfcells, :halfcells, :halfcells] = 0.796875;
        u0[:halfcells, :halfcells, :halfcells] = -0.1;
        v0[:halfcells, :halfcells, :halfcells] = 0.15;
        w0[:halfcells, :halfcells, :halfcells] = 0.094089;
        p0[:halfcells, :halfcells, :halfcells] = 0.4;

        #octant 8: y < 0.5, x > 0.5, z < 0.5
        r0[:halfcells, halfcells:, :halfcells] = 0.519705;
        u0[:halfcells, halfcells:, :halfcells] = -0.1;
        v0[:halfcells, halfcells:, :halfcells] = -0.15;
        w0[:halfcells, halfcells:, :halfcells] = 0.474136;
        p0[:halfcells, halfcells:, :halfcells] = 0.4;

    if con == 5: #12 contacts
        #octant 1: y > 0.5, x > 0.5, z > 0.5
        r0[halfcells:, halfcells:, halfcells:] = 0.5;
        u0[halfcells:, halfcells:, halfcells:] = -0.25;
        v0[halfcells:, halfcells:, halfcells:] = -0.5;
        w0[halfcells:, halfcells:, halfcells:] = -0.5;
        p0[halfcells:, halfcells:, halfcells:] = 1.0;

        #octant 2: y > 0.5, x < 0.5, z > 0.5
        r0[halfcells:, :halfcells, halfcells:] = 2.0;
        u0[halfcells:, :halfcells, halfcells:] = -0.25;
        v0[halfcells:, :halfcells, halfcells:] = 0.5;
        w0[halfcells:, :halfcells, halfcells:] = -0.25;
        p0[halfcells:, :halfcells, halfcells:] = 1.0;

        #octant 3: y < 0.5, x < 0.5, z > 0.5
        r0[:halfcells, :halfcells, halfcells:] = 0.5;
        u0[:halfcells, :halfcells, halfcells:] = 0.25;
        v0[:halfcells, :halfcells, halfcells:] = 0.5;
        w0[:halfcells, :halfcells, halfcells:] = 0.25;
        p0[:halfcells, :halfcells, halfcells:] = 1.0;

        #octant 4: y < 0.5, x > 0.5, z > 0.5
        r0[:halfcells, halfcells:, halfcells:] = 1.0;
        u0[:halfcells, halfcells:, halfcells:] = 0.25;
        v0[:halfcells, halfcells:, halfcells:] = -0.5;
        w0[:halfcells, halfcells:, halfcells:] = -0.25;
        p0[:halfcells, halfcells:, halfcells:] = 1.0;

        #octant 5: y > 0.5, x > 0.5, z < 0.5
        r0[halfcells:, halfcells:, :halfcells] = 1.0;
        u0[halfcells:, halfcells:, :halfcells] = 0.25;
        v0[halfcells:, halfcells:, :halfcells] = -0.25;
        w0[halfcells:, halfcells:, :halfcells] = -0.5;
        p0[halfcells:, halfcells:, :halfcells] = 1.0;

        #octant 6: y > 0.5, x < 0.5, z < 0.5
        r0[halfcells:, :halfcells, :halfcells] = 0.5;
        u0[halfcells:, :halfcells, :halfcells] = 0.25;
        v0[halfcells:, :halfcells, :halfcells] = 0.25;
        w0[halfcells:, :halfcells, :halfcells] = -0.25;
        p0[halfcells:, :halfcells, :halfcells] = 1.0;

        #octant 7: y < 0.5, x < 0.5, z < 0.5
        r0[:halfcells, :halfcells, :halfcells] = 2.0;
        u0[:halfcells, :halfcells, :halfcells] = -0.25;
        v0[:halfcells, :halfcells, :halfcells] = 0.25;
        w0[:halfcells, :halfcells, :halfcells] = 0.25;
        p0[:halfcells, :halfcells, :halfcells] = 1.0;

        #octant 8: y < 0.5, x > 0.5, z < 0.5
        r0[:halfcells, halfcells:, :halfcells] = 0.5;
        u0[:halfcells, halfcells:, :halfcells] = -0.25;
        v0[:halfcells, halfcells:, :halfcells] = -0.25;
        w0[:halfcells, halfcells:, :halfcells] = -0.25;
        p0[:halfcells, halfcells:, :halfcells] = 1.0;

    if con == 6: #6 shocks #6 contacts
        #octant 1: y > 0.5, x > 0.5, z > 0.5
        r0[halfcells:, halfcells:, halfcells:] = 0.53125;
        u0[halfcells:, halfcells:, halfcells:] = 0.00000;
        v0[halfcells:, halfcells:, halfcells:] = 0.00000;
        w0[halfcells:, halfcells:, halfcells:] = 0.00000;
        p0[halfcells:, halfcells:, halfcells:] = 0.40000;

        #octant 2: y > 0.5, x < 0.5, z > 0.5
        r0[halfcells:, :halfcells, halfcells:] = 1.00000;
        u0[halfcells:, :halfcells, halfcells:] = 0.727606875;
        v0[halfcells:, :halfcells, halfcells:] = 0.00000;
        w0[halfcells:, :halfcells, halfcells:] = 0.00000;
        p0[halfcells:, :halfcells, halfcells:] = 1.00000;

        #octant 3: y < 0.5, x < 0.5, z > 0.5
        r0[:halfcells, :halfcells, halfcells:] = 0.80000;
        u0[:halfcells, :halfcells, halfcells:] = 0.00000;
        v0[:halfcells, :halfcells, halfcells:] = 0.00000;
        w0[:halfcells, :halfcells, halfcells:] = -0.727606875;
        p0[:halfcells, :halfcells, halfcells:] = 1.00000;

        #octant 4: y < 0.5, x > 0.5, z > 0.5
        r0[:halfcells, halfcells:, halfcells:] = 1.00000;
        u0[:halfcells, halfcells:, halfcells:] = 0.00000;
        v0[:halfcells, halfcells:, halfcells:] = 0.727606875;
        w0[:halfcells, halfcells:, halfcells:] = 0.00000;
        p0[:halfcells, halfcells:, halfcells:] = 1.00000;

        #octant 5: y > 0.5, x > 0.5, z < 0.5
        r0[halfcells:, halfcells:, :halfcells] = 1.00000;
        u0[halfcells:, halfcells:, :halfcells] = 0.00000;
        v0[halfcells:, halfcells:, :halfcells] = 0.00000;
        w0[halfcells:, halfcells:, :halfcells] = 0.727606875;
        p0[halfcells:, halfcells:, :halfcells] = 1.00000;

        #octant 6: y > 0.5, x < 0.5, z < 0.5
        r0[halfcells:, :halfcells, :halfcells] = 0.80000;
        u0[halfcells:, :halfcells, :halfcells] = 0.00000;
        v0[halfcells:, :halfcells, :halfcells] = -0.727606875;
        w0[halfcells:, :halfcells, :halfcells] = 0.00000;
        p0[halfcells:, :halfcells, :halfcells] = 1.00000;

        #octant 7: y < 0.5, x < 0.5, z < 0.5
        r0[:halfcells, :halfcells, :halfcells] = 1.016216216;
        u0[:halfcells, :halfcells, :halfcells] = -0.401442839;
        v0[:halfcells, :halfcells, :halfcells] = -0.401442839;
        w0[:halfcells, :halfcells, :halfcells] = -0.401442839;
        p0[:halfcells, :halfcells, :halfcells] = 1.40000;

        #octant 8: y < 0.5, x > 0.5, z < 0.5
        r0[:halfcells, halfcells:, :halfcells] = 0.80000;
        u0[:halfcells, halfcells:, :halfcells] = -0.727606875;
        v0[:halfcells, halfcells:, :halfcells] = 0.00000;
        w0[:halfcells, halfcells:, :halfcells] = 0.00000;
        p0[:halfcells, halfcells:, :halfcells] = 1.00000;


    return r0, u0, v0, w0, p0, nx, ny, nz

import cupy as np
gamma = 1.4

con = 8
nxyz = 64
r0, u0, v0, w0, p0, nx, ny, nz = initialize(min = 0, max = 1, n = nxyz, con = con)
E0 = p0/((gamma-1.)*r0)+0.5*(u0**2 + v0**2 + w0**2) 
U0 = get_U_From_Cons(r0, r0*u0, r0*v0, r0*w0, r0*E0)
rho, rho_u, rho_v, rho_w, rho_E = get_Cons_From_U(U0, nx, ny, nz)

ovar = computeOtherVariables(rho, rho_u, rho_v, rho_w, rho_E)
a = ovar['a']
amax = np.max(np.absolute(a))
dy = (1/(ny-1))
dx = (1/(nx-1))
dz = (1/(nz-1))

tend=  0.3
dt = 0.0005

field = [r0, u0, v0, w0, E0]

seq_array = np.zeros((1, 5, ny, nx, nz))
t = 0
n = 0
while t < tend:

  flux_diff_y, flux_diff_x, flux_diff_z = modelfun(U0,nx,ny,nz)
  k1 = -dt/dx * (flux_diff_x) - dt/dy * (flux_diff_y) - dt/dz*(flux_diff_z)

  U0 = U0 + k1 
  rho, rho_u, rho_v, rho_w, rho_E = get_Cons_From_U(U0,nx,ny,nz)
  outputint = computeOtherVariables(rho, rho_u, rho_v, rho_w, rho_E)

  t += dt
  n += 1

  posit = outputint['E'] - 0.5*(outputint['u']*outputint['u'] + outputint['v']*outputint['v'] + outputint['w']*outputint['w'])
  field = np.array([outputint['r'], outputint['u'], outputint['v'], outputint['w'], posit])
  seq_array = np.append(seq_array, np.reshape(field, (1, 5, ny, nx, nz)), axis = 0)

import numpy as np
seq_array = np.delete(seq_array.get(), [0], axis = 0)
np.save('/path/field_con_' + str(con) + '_nxyz_' + str(nxyz) + '_dt_' + str(dt) + '_tend_' + str(tend) + '.npy', seq_array)

rho, rho_u, rho_v, rho_w, rho_E = get_Cons_From_U(U0,nx,ny,nz)
output = computeOtherVariables(rho, rho_u, rho_v, rho_w, rho_E)
u,v,w,p = output['u'], output['v'], output['w'], output['p']
