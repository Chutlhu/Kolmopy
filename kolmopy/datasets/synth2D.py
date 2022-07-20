"""
Data From Thomas Corpetti
"""

import os
from pathlib import Path
from time import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio

from tqdm import tqdm
from parse import parse

root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]

from kolmopy.utils import dwn_utils
from kolmopy.core import Dataset
from kolmopy.convention import _check_ij_indexing

BIBTEX = """
@misc{}
"""

DATASET_NAME = 'Synth2D'

REMOTES = {}

LICENSE_INFO = """
TO BE ADDED
"""

class Synth2D(Dataset):
    """
    Class to load Turb2D dataset.
    The data are 
    - t  : (times,) -> (T,)
    - xy : (times,res,res,xy) -> (T,512,512,2)
    - uv : (times,res,res,uv) -> (T,512,512,2)
    """

    def __init__(self, data_home):

        super().__init__(
            data_home,
            name=DATASET_NAME,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license=LICENSE_INFO
        )
        self.path_to_hdf5_file = self.data_home
        assert self.path_to_hdf5_file.exists()
        self.dset = h5py.File(self.path_to_hdf5_file.resolve(),'r')
        
        self.nt = None
        self.nx = None
        self.ny = None
        self.data = None
        self.delta_t = None
    
        self.indexing = 'ij'
        self.img_res = [512,512]

        self.diffusion = self.dset.attrs['diffusion']

        self.variables = {'txy':None}
        self.fields = {'uvw':None}
        pass

    @property
    def t(self):       
        return self.variables['txy'][:,0,0,0].squeeze()
    @property
    def x(self): return self.variables['txy'][0,:,0,1].squeeze()
    @property
    def y(self): return self.variables['txy'][0,0,:,2].squeeze()
    @property
    def xy(self): return self.variables['txy'][0,:,:,1:3].squeeze()
    @property
    def txy(self): return self.variables['txy'].squeeze()
    @property
    def u(self): return self.fields['uvw'][...,0].squeeze()
    @property
    def v(self): return self.fields['uvw'][...,1].squeeze()
    @property
    def uv(self): return self.fields['uvw'][...,:2].squeeze()
    @property
    def w(self): return self.fields['uvw'][...,2].squeeze()
    @property
    def uvw(self): return self.fields['uvw'].squeeze()
    @property
    def fshape(self): return self.fields['uvw'].shape
    @property
    def vshape(self): return self.variables['txy'].shape
    
    def validate(self):
        """
        pretty much hardcoded to check that the publishing was correct
        """
        print('Check data dimensions... ', end='')
        self.load_data(time_idx=np.arange(5), only_vel_field=False)
        try:
            assert self.txy.shape == self.uvw.shape
            assert _check_ij_indexing(self.xy)
        except:
            raise ValueError('Something went wrong. Possibly corrupted dataset.')
        print('done!')

    def load_data(self, time_idx=None, only_vel_field=True):
        if not isinstance(time_idx, np.ndarray) and not time_idx is None:
            time_idx = np.array([time_idx])


        if only_vel_field:
            self.variables = {
                'txy' : self.get_coords(time_idx)
            }
            self.fields = {
                'uvw' : self.get_fields(time_idx),
            }

        else:
            self.variables = {
                'txy' : self.get_coords(time_idx)}
            self.fields = {
                'uvw' : self.get_fields(time_idx),
            }
        return
        
    def get_time(self, time_idx=None):
        if time_idx is None:
            return self.dset['txy_vars'][:, 0, 0, 0]
        else:
            return self.dset['txy_vars'][time_idx, 0, 0, 0]

    def get_coords(self, time_idx=None):
        if time_idx is None:
            return self.dset['txy_vars'][:, :, :, :3]
        else:
            return self.dset['txy_vars'][time_idx, :, :, :3]
        
    def get_fields(self, time_idx=None):
        if time_idx is None:
            return self.dset['txy_vars'][:, :, :, 3:]
        else:
            return self.dset['txy_vars'][time_idx, :, :, 3:]

    def close(self,):
        return self.dset.close()

    def __str__(self) -> str:
        name = self.name
        data_path = self.data_home
        return f'{name} stored in {data_path}.\n- variables {self.variables.keys()}, dim {self.vshape}\n- fields: {self.fields.keys()} dim {self.fshape} '


def from_raw_data_to_hdf5(dset, dset_name, mat_file):
    
    print('Opening matfile...')
    matfile = sio.loadmat(mat_file)
    print(' Done!')

    Ui = matfile['Uinter'].squeeze()
    Vi = matfile['Vinter'].squeeze()
    Uf = matfile['Ufinal'].squeeze()
    Vf = matfile['Vfinal'].squeeze()
    Wi = matfile['vorticite_inter'].squeeze()
    Wf = matfile['vorticite_final'].squeeze()
    
    diffusion = matfile['diffusion'].squeeze()
    dset.attrs['diffusion'] = diffusion

    nb_steps = matfile['nb_steps'].squeeze()
    nb_steps = nb_steps.tolist()
    nb_steps = np.array([0]+nb_steps) # need to add the starting one

    Dt = 1
    Nt = len(nb_steps)
    assert Nt == Uf.shape[-1]
    Tf = np.arange(Nt)*Dt

    n_vars = 6 # x,y,t,u,v,w
    Ti = []
    for i in range(Nt-1):
        curr_Ti = np.linspace(Tf[i], Tf[i+1], nb_steps[i+1], endpoint=False)
        Ti.append(curr_Ti)
    Ti.append([Tf[-1]])
    Ti = np.concatenate(Ti)
    
    Rx = Ui.shape[0]
    Ry = Ui.shape[1]
    assert Rx == Ry
    Rt = len(Ti)
    
    data = np.zeros((Rt, Rx, Ry, n_vars)) # add time

    # /!\ Warning /!\
    #  yv and xv swapped for ij convention
    yv, xv, tv = np.meshgrid(np.linspace(-1,1,Rx), np.linspace(-1,1,Ry), Ti)


    # ensure convention
    _check_ij_indexing(np.stack([xv[:,:,0], yv[:,:,1]], axis=-1))

    for t in tqdm(range(Rt)):

        data[t,:,:,0] = tv[:,:,t]
        data[t,:,:,1] = xv[:,:,t]
        data[t,:,:,2] = yv[:,:,t]
        data[t,:,:,3] = Vi[:,:,t]   # /!\ Warning /!\ ij convention
        data[t,:,:,4] = Ui[:,:,t]   # /!\ Warning /!\ ij convention
        data[t,:,:,5] = Wi[:,:,t]

        ts = Ti[t]

        assert np.unique(tv[:,:,t]) == ts

        if np.mod(ts,1) == 0:
            T = int(ts)
            assert np.allclose(Ti[t],     Tf[T])
            assert np.allclose(Ui[:,:,t], Uf[:,:,T])
            assert np.allclose(Vi[:,:,t], Vf[:,:,T])
            assert np.allclose(Wi[:,:,t], Wf[:,:,T])

    dset.create_dataset(dset_name, data=data, compression="gzip", compression_opts=5)
    return dset

def publish_as_hdf5(mat_file, output_data_hdf5):
    """Pretty much hardcoded"""

    dset = h5py.File(output_data_hdf5,'w')            
    # try:
    from_raw_data_to_hdf5(dset, 'txy_vars', mat_file)
    # except Exception as exc:
    #     print('ERROR:', exc)
    dset.close()


if __name__ == '__main__':
    """
    example:
    >>> python synth2D.py path/to/dataset/folder -o ./.cached/

    """
    import argparse
    parser = argparse.ArgumentParser("Synth2D.publish",
                                     description="Publish Synth2D to HDF5")
    parser.add_argument("--file", type=Path, 
                        help='Path to original file')
    parser.add_argument("-o", "--out", type=Path,
                        default=Path(".cached"),
                        help="Folder where to put hdf5 dataset.")
    args = parser.parse_args()
    input_file = Path(args.file)
    output_data_dir = Path(args.out)

    # parse input
    str_format = 'all_fields_{id}.mat'
    params = parse(str_format, input_file.name)
    simul_id = params['id']

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f'{DATASET_NAME}_{simul_id}.hdf5'
    print(f"HDF5 Dataset will be stored in: {out_file.resolve()}")
    publish_as_hdf5(input_file, out_file)
    print("So long, and tank you for the fish.")

