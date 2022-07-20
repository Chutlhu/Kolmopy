import numpy as np
from PIL import Image, ImageDraw

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def draw_circle(img_dim, r):
    Rx, Ry = img_dim
    #size of circle
    e_x, e_y = r, r
    #create a box 
    bbox=((Rx/2)-(e_x/2), (Ry/2)-(e_y/2), (Rx/2)+(e_x/2),(Ry/2)+(e_y/2))
    img = Image.new("L",(Rx, Ry),color=0)
    draw1 = ImageDraw.Draw(img)
    draw1.ellipse(bbox, fill=255)
    return np.array(img)