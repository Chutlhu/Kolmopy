import torch
import numpy as np
import matplotlib.pyplot as plt

from kolmopy.convention import _check_dims_img, _check_ij_indexing

"""
Sources
- https://github.com/maxjiang93/space_time_pde
- https://github.com/Rose-STL-Lab/Turbulent-Flow-Net
- https://github.com/b-fg/Energy_spectra
"""


def my_grad(f:tuple, sp:tuple, indexing:str = "xy", order=1):
    """
    my local computation of the gradient
    
    f: tuple 
        -> vector field components [Fx,Fy,Fz,...]
    sp: tuple 
        -> spacing between points in respecitve directions [spx, spy,spz,...]
        -> or 1N array for the coordinates [x,y,z,...]
    indexing: str 
        "xy" or "ij", see np.meshgrid indexing 
    
    Returns:
        Components x Directions: grad[j][i] is the gradient of j-th component of F with respect direction i
    """
    num_dims_f = len(f)
    num_dims_x = len(sp)

    if indexing == "xy":
        raise NotImplementedError
        # return [[np.gradient(f[num_dims - j - 1], sp[i], axis=i, edge_order=1) 
        #         for i in range(num_dims)] for j in range(num_dims)]
    if indexing == "ij":
        return [[np.gradient(f[j], sp[i], axis=i, edge_order=order) 
                for i in range(num_dims_x)] for j in range(num_dims_f)]


def compute_vorticity(xy, uv, indexing='ij', order=1):
    _check_ij_indexing(xy)
    _check_dims_img(xy)
    _check_dims_img(uv)
    assert xy.shape == uv.shape

    x = xy[:,0,0]
    y = xy[0,:,1]
    du_xy = my_grad([uv[:,:,0], uv[:,:,1]], [x, y], indexing=indexing, order=order)
    return du_xy[1][0] - du_xy[0][1]


def compute_divergence(xy, uv, indexing='ij', order=1):
    _check_ij_indexing(xy)
    _check_dims_img(xy)
    _check_dims_img(uv)
    assert xy.shape == uv.shape

    x = xy[:,0,0]
    y = xy[0,:,1]
    du_xy = my_grad([uv[:,:,0], uv[:,:,1]], [x, y], indexing=indexing, order=order)
    return du_xy[0][0] + du_xy[1][1]


def compute_magnitude(uv):
    _check_dims_img(uv)
    return np.sqrt(uv[:,:,0]**2 + uv[:,:,1]**2)


def powerspec(uv):
    """
    By Thomas Corpetti
    """
    _check_dims_img(uv)

    Rx, Ry, D = uv.shape

    eps = 1e-50 # to avoid log(0)
    c  = np.sqrt(1.4)
    Ma = 0.1
    U0 = Ma*c
    uv = uv/U0
    dims = Rx * Ry
    
    amplsU = np.abs(np.fft.fft2(uv[:,:,0])/dims)
    amplsV = np.abs(np.fft.fft2(uv[:,:,1])/dims)

    EK_U  = 0.5*(amplsU**2 + amplsV**2)

    EK_U = np.fft.fftshift(EK_U)

    sign_sizex = np.shape(EK_U)[0]
    sign_sizey = np.shape(EK_U)[1]

    box_sidex = sign_sizex
    box_sidey = sign_sizey

    box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)

    centerx = int(box_sidex/2)
    centery = int(box_sidey/2)
                
    EK_U_avsphr = np.zeros(box_radius,)+eps ## size of the radius
    
    for i in range(box_sidex):
        for j in range(box_sidey):
            wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2)))
            EK_U_avsphr[wn] = EK_U_avsphr[wn] + EK_U[i,j]

    k_bin = np.arange(0,box_radius)
    return EK_U_avsphr, k_bin


def _my_spec_grad(S):
    """
    Compute spectral gradient of a scalar field
    """
    assert(len(S.shape) in [3])
    res = S.shape[-1]
    # compute the frequency support
    k = torch.fft.fftfreq(res, d=1/res, device=S.device)
    k = torch.stack(torch.meshgrid([k, k]), dim=0)
    # compute gradient (dS = j*w*S)
    return 1j*k*S

def fluct(uv):
    _check_dims_img(uv)
    return uv - torch.mean(uv, dim=[0,1])

def tkenergy(uv):
    _check_dims_img(uv)
    # tke = 0.5 * (uv[0,:,:]**2 + uv[1,:,:]**2)
    tke = 0.5 * torch.mean(uv**2, dim=0)
    return torch.mean(tke)


def dissipation(uv, viscosity):
    _check_dims_img(uv)
    UV = torch.fft.fft2(uv) # (2, Rx, Ry)
    dUV = _my_spec_grad(UV) # (2, Rx, Rx)
    dUVt = dUV.permute(0,2,1) # (2, Ry, Rx)
    # compute (spectral) strain
    S = 0.5 * (dUV + dUVt)
    s = torch.fft.ifft2(uv).real
    diss = 2 * viscosity * torch.mean(s**2, dim=0)
    return torch.mean(diss)


def rmsvelocity(uv):
    _check_dims_img(uv)
    return tkenergy(uv) * (2./3.)**(1./2.)


def tmscale(uv, viscosity):
    rmvs = rmsvelocity(uv)
    diss = dissipation(uv, viscosity=viscosity)
    return (15*viscosity*(rmvs**2)/diss)**(1/2)


def tsreynolds(uv, viscosity):
    rmsv = rmsvelocity(uv)
    lam = tmscale(uv, viscosity)
    return rmsv * lam / viscosity


def ktimescale(uv, viscosity):
    diss = dissipation(uv, viscosity)
    return (viscosity/diss)**(1./2.)


def klenscale(uv, viscosity):
    diss = dissipation(uv, viscosity)
    return viscosity**(3/4) * diss**(-1/4)


def intscale(uv):
    UV, k = energy_spectrum(uv)
    rmsv = rmsvelocity(uv)

    c1 = np.pi/(2*rmsv**2)
    c2 = torch.sum(UV / k, dim=0)
    return c1 * c2
