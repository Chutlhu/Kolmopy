"""
D. Heitz 11/06/2021

Dataset Images with synthetic particles and scalar based on DNS of self sustained 2D turbulence.
Reference coordinate system is bottom left with x,dx in horizontal-right direction and y,dy in vertical-top direction
dx and dy are displacements between two following images in pixel.

D. Heitz, J. Carlier, and G. Arroyo.
Final report on the evaluation of the tasks of the workpackage 2, FLUID project deliverable 5.4. 
Technical report, 2007.


"""

import os
from pathlib import Path
from wsgiref import validate
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from natsort import natsorted

root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]

from kolmopy.utils import dwn_utils
from kolmopy.core import Dataset

BIBTEX = """
@misc{heitz2007deliverable,
  title={Final report on the evaluation of the tasks of the workpackage 2, FLUID project deliverable 5.4.},
  author={Heitz, Dominique and Carlier, Johan and Arroyo, Georges and Szantai, Andr{\'e}}
  year={2007},
}
"""

# prototype for downloading raw data from the internet
# REMOTES = {
#     "dataset": dwn_utils.RemoteFileMetadata(
#         filename="FSDnoisy18k.audio_train.zip",
#         url="https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_train.zip?download=1",
#         checksum="34dc1d34ca44622af5bf439ceb6f0d55",
#     ),
# }
REMOTES = {}

LICENSE_INFO = """
TO BE ADDED
"""

class Turb2D(Dataset):
    """
    Class to load Turb2D dataset.
    The data are 
    - t  : (times,) -> (1000,)
    - xy : (times,res,res,xy) -> (1000,256,256,2)
    - uv : (times,res,res,uv) -> (1000,256,256,2)
    """

    def __init__(self, data_home):

        super().__init__(
            data_home,
            name='Turb2D',
            bibtex=BIBTEX,
            remotes=REMOTES,
            license=LICENSE_INFO
        )
        self.path_to_hdf5_file = self.data_home
        assert self.path_to_hdf5_file.exists()
        self.dset = h5py.File(self.path_to_hdf5_file.resolve(),'r')
        
        self.nt = 1000
        self.nx = 256
        self.ny = 256
        
        self.data = None

        self.delta_t = 0.01
        self.indexing = 'ij'
        self.times = self.delta_t*np.arange(self.nt)
        self.min_dx = 0.00784314
        self.min_dy = 0.00784314

        self.variables = {'xy':None, 't':None}
        self.fields = {'uv':None, 'particle':None, 'scalar':None}
        pass

    @property
    def t(self): return self.variables['t']
    @property
    def x(self): return self.variables['xy'][...,0]
    @property
    def y(self): return self.variables['xy'][...,1]
    @property
    def xy(self): return self.variables['xy']
    @property
    def u(self): return self.fields['uv'][...,0]
    @property
    def v(self): return self.fields['uv'][...,1]
    @property
    def uv(self): return self.fields['uv']
    @property
    def scalar(self): return self.fields['scalar']
    @property
    def particle(self): return self.fields['particle']
    
    def validate(self):
        """
        pretty much hardcoded to check that the publishing was correct
        """
        t = np.arange(self.nt)
        print('Check data dimensions... ', end='')
        self.load_data(time_idx=t, only_vel_field=False)

        try:
            assert len(self.t) == self.nt
            assert self.xy.shape == (self.nt, self.nx, self.ny, 2)
            assert self.uv.shape == (self.nt, self.nx, self.ny, 2)
            assert self.scalar.shape == (self.nt, self.nx, self.ny)
            assert self.particle.shape == (self.nt, self.nx, self.ny)
        except:
            raise ValueError('Something went wrong. Possibly corrupted dataset.')
        print('done!')

    def load_data(self, time_idx=None, only_vel_field=True):
        if only_vel_field:
            self.variables = {
                't'  : self.get_time(time_idx),
                'xy' : self.get_coords(time_idx)
            }
            self.fields = {
                'uv' : self.get_velocity(time_idx),
            }

        else:
            self.variables = {
                't'  : self.get_time(time_idx),
                'xy' : self.get_coords(time_idx)}
            self.fields = {
                'uv' : self.get_velocity(time_idx),
                'scalar' :  self.get_scalar(time_idx),
                'particle' : self.get_particle(time_idx),
            }
        return
        
    def get_time(self, time_idx=None):
        if time_idx is None:
            return self.times
        else:
            return self.times[time_idx]

    def get_coords(self, time_idx=None):
        if time_idx is None:
            return self.dset['velocity'][:, :, :, :2]
        else:
            return self.dset['velocity'][time_idx, :, :, :2].squeeze()
        
    def get_velocity(self, time_idx=None):
        if time_idx is None:
            return self.dset['velocity'][:, :, :, 2:]
        else:
            return self.dset['velocity'][time_idx, :, :, 2:].squeeze()
        
    def get_scalar(self, time_idx=None):
        if time_idx is None:
            return self.dset['scalar'][()]
        else:
            return self.dset['scalar'][time_idx, :, :, :].squeeze()
        
    def get_particle(self, time_idx=None):
        if time_idx is None:
            return self.dset['particle'][()]
        else:
            return self.dset['particle'][time_idx, :, :, :].squeeze()

    def close(self,):
        return self.dset.close()


    def __str__(self) -> str:
        name = self.name
        data_path = self.data_home
        return f'{name} stored in {data_path}.\n- variables {self.variables.keys()}\n- fields: {self.fields.keys()} '

def from_raw_data_to_hdf5(dset, dset_name, data_dir, ext, img_dims, n_vars):
    
    files =  [f for f in data_dir.glob(f'*.{ext}')]
    files = natsorted(files)
    n_files = len(files)

    data = np.zeros((n_files, *img_dims, n_vars)) # add time

    for f, file in enumerate(tqdm(files)):
        if ext == 'txt':
            curr_data = np.loadtxt(file.open())
            curr_data[:, :2] = curr_data[:, :2]/255

        if ext == 'tif':
            curr_data = plt.imread(file)
            curr_data = curr_data / 255

        data[f, ...] = curr_data.reshape(*img_dims, n_vars)

    dset.create_dataset(dset_name, data=data, compression="gzip", compression_opts=5)


def publish_as_hdf5(input_data_dir, output_data_hdf5):
    """Pretty much hardcoded"""

    raw_data_info = {
        'velocity'  : {
            'raw_data_dir' : Path(input_data_dir, 'shifttruth1000-txt.txt'),
            'raw_data_ext' : 'txt',
            'raw_data_nvars' : 4,
            'raw_data_img_dims' : (256, 256),
        },
        'scalar'  : {
            'raw_data_dir' : Path(input_data_dir, 'scalar'),
            'raw_data_ext' : 'tif',
            'raw_data_nvars' : 1,
            'raw_data_img_dims' : (256, 256),
        },
        'particle'  : {
            'raw_data_dir' : Path(input_data_dir, 'particle'),
            'raw_data_ext' : 'tif',
            'raw_data_nvars' : 1,
            'raw_data_img_dims' : (256, 256),
        },
    }

    dset = h5py.File(output_data_hdf5,'w')
    try:
        for key in raw_data_info:
            data_dict = raw_data_info[key]
            print(f'Creating {key}: from %s in hdf5' % data_dict['raw_data_ext'])
            
            from_raw_data_to_hdf5(
                dset, key,
                data_dict['raw_data_dir'], 
                data_dict['raw_data_ext'], 
                data_dict['raw_data_img_dims'], 
                data_dict['raw_data_nvars'],
                )
    except Exception as exc:
        print(exc)
    dset.close()


if __name__ == '__main__':
    """
    example:
    >>> python turb2D.py path/to/dataset/folder -o ./.cached/

    """
    import argparse
    parser = argparse.ArgumentParser("turb2D.publish",
                                     description="Publish turb2D to HDF5")
    parser.add_argument("data_home", type=Path, 
                        help='Path to original data')
    parser.add_argument("-o", "--out", type=Path,
                        default=Path(".cached"),
                        help="Folder where to put hdf5 dataset.")
    args = parser.parse_args()
    input_data_dir = Path(args.data_home)
    output_data_dir = Path(args.out)
    
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / 'Turb2D.hdf5'
    print(f"HDF5 Dataset will be stored in: {out_file.resolve()}")
    publish_as_hdf5(input_data_dir, out_file)
    print("So long, and tank you for the fish.")

