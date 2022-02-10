# The Kolmopy convention
#
# For meshgrid data:
# (Time x) [D1 x D2 x ... x D3 ] x Nvars
# for instance
# - :single image:        256 x 256 x 2 for (x, y)  ->        256 x 256 x 2 for (u, v)
# - :multip image: 1000 x 256 x 256 x 3 for (t,x,y) -> 1000 x 256 x 256 x 2 for (t,u,v)

# for vectorial data:
# Nsamples x Nvars

def _check_dims_img(uv):
    # variables and fields must be in the shape of:
    # - (H,W,2)
    # - (T, H, W, 2)
    if not len(uv.shape[:-1]) == uv.shape[-1]:
        raise ValueError(f'Wrong input dimension, got {uv.shape}')
    

def _check_dim_vect(uv):
    # check that uv is (R*R,2)
    if not len(uv.shape) == 2:
        raise ValueError(f'N dimension should be 2, got {uv.shape}')