from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

import kolmopy.utils.phy_utils as phy
from kolmopy.convention import _check_dim_vect, _check_dims_img

###############################################################################
##                                 GENERAL UTIS                              ##
###############################################################################

def _validate_dims(xy=None, uv=None, kind='img'):

    try:
        if not xy is None:
            _check_dims_img(xy)

        if not uv is None:
            _check_dims_img(uv)
        
        if not xy is None and not uv is None:
            assert xy.shape == uv.shape
    
    except Exception as exc:
        print(exc)
        raise ValueError('Please respect the xy and uv convenction.')


###############################################################################
##                                 SIMPLE PLOTS                              ##
###############################################################################

def plot_2D_velocity_field(xy, uv, step=5, scale=20, ax=None, bg_img=None, bg_img_indexing='ij'):
    """
    Created on Tue Sep 15 13:22:23 2015
    @author: corpetti

    affichage d'un champ de vecteurs
    Input : u,v,scale,step,image (3 derniers optionnels)
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    s = step

    _validate_dims(xy, uv)

    if not bg_img is None:
        # # we need to transpose the image since field is computed with 
        if bg_img_indexing == 'ij':
            w = bg_img.T[::-1,:]
        if bg_img_indexing == 'xy':
            w = bg_img
        # w = np.zeros_like(background_img)
        # for i in range(w.shape[0]):
        #     for j in range(w.shape[1]):
        #         w[i,j] = background_img[j, -i]

        xmin, xmax = np.min(xy[::s,::s,0]), np.max(xy[::s,::s,0])
        ymin, ymax = np.min(xy[::s,::s,1]), np.max(xy[::s,::s,1])
        ax.imshow(w,
                    extent=[xmin, xmax, ymin, ymax], 
                    origin=None, cmap='gray')
    
    color = np.sqrt((uv[:,:,0]**2) + (uv[:,:,1]**2))
    qr = ax.quiver(
            xy[::s,::s,0],
            xy[::s,::s,1], 
            uv[::s,::s,0],
            uv[::s,::s,1], 
            color[::s,::s], 
            scale=scale)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return qr


###############################################################################
##                              SIMPLE ANIMATIONS                            ##
###############################################################################

def animate_image(img_t, interval=100, repeat_delay=1000, save_path=None):

    assert len(img_t.shape) == 3

    T = img_t.shape[0]

    fig, axarr = plt.subplots(figsize=(10,10))

    ims = []
    for t in tqdm(range(T)):
        im = axarr.imshow(img_t[t,:,:], animated=True)
        ims.append([im])
        
    anim = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=repeat_delay)
    plt.close()

    if save_path is not None:
        assert save_path.split('.')[-1] == 'mp4'
        my_file = Path(save_path)
        writervideo = animation.FFMpegWriter(fps=30)
        anim.save(my_file, writer=writervideo)

    return anim


def animate_2D_velocity_field(xy_t, uv_t, show_vorticity=False, step=5, scale=20, indexing='ij', save_path=None):
    
    _validate_dims(xy_t[0,...], uv_t[0,...])
    T = xy_t.shape[0]

    fig, ax = plt.subplots(figsize=(5,5))
    ims = []

    for t in tqdm(range(T)):

        xy = xy_t[t,...]
        uv = uv_t[t,...]

        im = ax.quiver(
            xy[::step,::step,0],
            xy[::step,::step,1], 
            uv[::step,::step,0],
            uv[::step,::step,1], 
            # color[::s,::s], 
            scale=scale, animated=True)

        ims.append([im])

    anim = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.close()

    if save_path is not None:
        assert save_path.split('.')[-1] == 'gif'
        my_file = Path(save_path)
        anim.save(my_file, metadata={'artist':'Guido'}) 


    return anim


###############################################################################
##                                OTHER PLOTS                                ##
###############################################################################

def plot_middlebury_colors(uv_img, ax=None):

    _validate_dims(uv=uv_img)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    img = flow_to_middlebury_rbg(uv_img)

    ax.imshow(img)

    return  ax


def flow_to_middlebury_rbg(uv_img):
    """
    from https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = uv_img[:,:,0]
    v = uv_img[:,:,1]

    maxu = -999.
    maxv = -999.
    minu =  999.
    minv =  999.

    idxUnknow = (abs(u) > 1e7) | (abs(v) > 1e7)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    from https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    from https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

