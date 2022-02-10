
import os
from pathlib import Path
import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.animation as ani
from numpy.lib.npyio import load

from tqdm import tqdm

from natsort import natsorted

import turboflow.utils.viz_utils as viz

root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]

class Re3900():
    def __init__(self, data_dir=None):

        if data_dir is None:
            data_dir = Path('/','home','dicarlo_d','Documents','Datasets','2021-Re3900')
        self.data_dir = Path(data_dir)

        # hardcoded
        self.name = 'Re3900'
        self.cache_dir = Path(root_dir, '.cache')
        self.cached_hdf5_data_file = Path(self.cache_dir, f'{self.name}.hdf5')

        self.nt = 1000
        self.nx = 308
        self.ny = 328
        self.nz = 3

        self.raw_data_info = {
            'velocity'  : {
                'raw_data_dir' : Path(data_dir, '2D_DNS_3900'),
                'raw_data_ext' : 'txt',
                'raw_data_nvars' : 6,
                'raw_data_img_dims' : (self.nx, self.ny),
            },
        }
        
        self.data = None

        self.delta_t = 0.01
        self.indexing = 'ij'
        self.t = self.delta_t*np.arange(self.nt)

        self.variables = {'xyz':None, 'uvw':None}
        pass

    @property
    def x(self): return self.variables['xyz'][...,0]
    @property
    def y(self): return self.variables['xyz'][...,1]
    @property
    def xy(self): return self.variables['xyz'][...,:2]
    @property
    def xyz(self): return self.variables['xyz']
    @property
    def u(self): return self.variables['uvw'][...,0]
    @property
    def v(self): return self.variables['uvw'][...,1]
    @property
    def uv(self): return self.variables['uvw'][...,:2]
    @property
    def uvw(self): return self.variables['uvw']
    @property
    def scalar(self): return self.variables['scalar']
    @property
    def particle(self): return self.variables['particle']
    

    def setup(self):
        # if not self.cached_hdf5_data_file.exists():
        for key in self.raw_data_info:
            data_dict = self.raw_data_info[key]
            print(f'Creating {key}: from %s in hdf5' % data_dict['raw_data_ext'])
            self._from_raw_data_to_hdf5(key,
                data_dict['raw_data_dir'], 
                data_dict['raw_data_ext'], 
                data_dict['raw_data_img_dims'], 
                data_dict['raw_data_nvars'],
            )

        self.dset = h5py.File(self.cached_hdf5_data_file,'r')
        return

    def load_data(self, time_idx=None, z_idx=None):
        self.variables = {
            't'  : self.get_time(time_idx),
            'xyz' : self.get_coords(time_idx, z_idx),
            'uvw' : self.get_velocity(time_idx, z_idx),
        }
        return self.variables


    def _from_raw_data_to_hdf5(self, dset_name, data_dir, ext, img_dims, n_vars):
        files =  [f for f in data_dir.glob(f'*.{ext}')]
        files = natsorted(files)
        n_files = len(files)
        assert n_files == self.nt

        data = np.zeros((n_files, *img_dims, self.nz, n_vars)) # add time

        for f, file in enumerate(tqdm(files)):

            if ext == 'txt':
                curr_data = np.loadtxt(file.open())
                curr_data = curr_data.reshape(self.nz, self.nx, self.ny, n_vars)
                curr_data = curr_data.transpose(1,2,0,3)

            data[f, ...] = curr_data
            
        dset = h5py.File(self.cached_hdf5_data_file,'w')
        dset.create_dataset(dset_name, data=data, compression="gzip", compression_opts=5)
        dset.close()
        
    def get_time(self, time_idx=None):
            return self.t[time_idx]

    def get_coords(self, time_idx=None, z_idx=None):
        return self.dset['velocity'][time_idx, :, :, z_idx, :3].squeeze()

    def get_velocity(self, time_idx=None, z_idx=None):
        return self.dset['velocity'][time_idx, :, :, z_idx, 3:].squeeze()


    def plot_data(self, time_idx=None):
        pass

if __name__ == '__main__':

    import pathlib
    data_dir = pathlib.Path('/','home','dicarlo_d','Documents','Datasets','2021-Re3900')
    Re3900 = Re3900(data_dir)
    Re3900.setup()