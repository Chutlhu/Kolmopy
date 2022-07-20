from asyncore import loop
from datetime import time
import glob
import numpy as np
from pathlib import Path
from natsort import natsorted

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt

import pytorch_lightning as pl

from kolmopy.datasets.turb2D import Turb2D
from kolmopy.datasets.re3900 import Re3900
from kolmopy.datasets.synth2D import Synth2D
from kolmopy.core import Dataset
import kolmopy.utils.phy_utils as phy

from kornia.filters import gaussian_blur2d, median_blur
from turboflow.utils.img_utils import draw_circle

currently_supported_dataset = ['Turb2D', 'Re39000', 'Synth2D']

class Re39000Dataset(Dataset):

    def __init__(self, data_dir:str, ds:int, time_idx:int, z=1):
        super(Re39000Dataset, self).__init__()
        
        self.name = "Re39000"

        self.time_idx = time_idx
        # load raw data
        files = natsorted(list(glob.glob(data_dir + "*.txt")))
        file = files[time_idx]
        data = np.loadtxt(file)

        self.size = data.shape[0]

        # hardcoded data processing
        self.n_vars = 6 # x, y, z, u, v, w
        self.nx = 308
        self.ny = 328
        self.nz = 3
        data = data.reshape(self.nz, self.nx, self.ny, self.n_vars)
        data = data.traspose(1,2,0,3)

        # hardcoded data downsampling
        #             x     y   z  vars
        data = data[::ds, ::ds, z, :][:,:,None,:]
        self.nx = data.shape[0]
        self.ny = data.shape[1]
        self.nz = data.shape[2]

        self.order_x = "xyz"
        self.order_y = "uvw"

        self.X = torch.from_numpy(data[...,:3]).float().view(-1, 2) # x, y, z
        self.y = torch.from_numpy(data[...,3:]).float().view(-1, 2) # u, v, w

        assert self.X.shape == self.y.shape

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        X = self.X[idx,:]
        y = self.y[idx,:]
        return X, y


class Turb2DDataset(Dataset):
    """
    Class to load Turb2D dataset in pytorch.
    The data follows the scikit-learn convenction: 
    
    if time is provided as array, it returns
    - X : (N, txy) -> (1000x256x256, 3)
    - y : (N, uv)  -> (1000x256x256, 2)

    if t is scalar, then it returns
    - X : (N, xy) -> (1000x256x256, 2)
    - u : (N, uv) -> (1000x256x256, 2)
    """

    def __init__(self,data_dir:str=None,ds:int=1,dt:int=1,time_idx:int=None):

        tb = Turb2D(data_dir)
        tb.load_data(time_idx)

        # Data in Turb2D are (T,R,R,D)
        t = tb.t
        X = tb.xy
        y = tb.uv

        # normalize y
        y = y/np.max(np.abs(y))
        assert np.min(y) >= -1
        assert np.max(y) <=  1

        assert X.shape == y.shape
        assert len(X.shape) in [3,4]

        if len(X.shape) == 3: # single image/time
            
            # downsampling
            X = X[::ds, ::ds, :]
            y = y[::ds, ::ds, :]
        
            self.res = X.shape[0] 
            self.img_shape = X.shape # (R,R,2)

            self.X = torch.from_numpy(X).float().view(-1,2)
            self.y = torch.from_numpy(y).float().view(-1,2)
            self.t = float(t)
            self.size = self.X.shape[0]

        if len(X.shape) == 4: # multiple images/times
            # downsampling
            X = X[::dt, ::ds, ::ds, :]
            y = y[::dt, ::ds, ::ds, :]
            t = t[::dt, None, None, None] * np.ones((X.shape[0], X.shape[1], X.shape[2], 1))
            
            X = np.concatenate([t, X], axis=-1)

            self.times = X.shape[0]
            self.res = X.shape[1] 
            self.img_res = X.shape[1:3] # (R,R,2)
            self.vars_shape_img = X.shape

            self.X = torch.from_numpy(X).float().view(-1,3)
            self.y = torch.from_numpy(y).float().view(-1,2)
            self.size = self.X.shape[0]

        assert self.X.shape[0] == self.y.shape[0]
                
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X = self.X[idx,:]
        y = self.y[idx,:]
        return X, y


class Synth2DDataset(Dataset):
    """
    Class to load Turb2D dataset in pytorch.
    The data follows the scikit-learn convenction: 
    
    if time is provided as array, it returns
    - X : (N, txy) -> (1000x256x256, 3)
    - y : (N, uv)  -> (1000x256x256, 2)

    if t is scalar, then it returns
    - X : (N, xy) -> (1000x256x256, 2)
    - u : (N, uv) -> (1000x256x256, 2)
    """

    def __init__(self,data_dir:str=None,ds:int=1,dt:int=1,time_idx:int=None):

        add_vorticity=True
        tb = Synth2D(data_dir)
        tb.load_data(time_idx)

        # Data in Turb2D are (T,R,R,D)
        t = tb.t
        X = tb.txy
        if add_vorticity:
            y = tb.uvw
        else:
            y = tb.uv

        # normalize y
        if add_vorticity:
            y[...,:2] = y[...,:2]/np.max(np.abs(y[...,:2]))
            y[...,-1] = y[...,-1]/np.max(np.abs(y[...,-1]))
        else:
            y = y/np.max(np.abs(y))
        assert np.min(y) >= -1
        assert np.max(y) <=  1

        assert len(X.shape) in [3,4]

        if len(X.shape) == 3: # single image/time
            
            # downsampling
            X = X[::ds, ::ds, :]
            y = y[::ds, ::ds, :]
        
            self.res = X.shape[0] 
            self.img_shape = X.shape # (R,R,2)

            self.X = torch.from_numpy(X).float().view(-1,2)
            self.y = torch.from_numpy(y).float().view(-1,2)
            self.t = float(t)
            self.size = self.X.shape[0]

        if len(X.shape) == 4: # multiple images/times
            # crop
            C = 0
            if C > 0:
                X = X[:, C:-C, C:-C, :]
                y = y[:, C:-C, C:-C, :]
            print(X.shape)
            print(y.shape)
            # downsampling
            X = X[::dt, ::ds, ::ds, :]
            y = y[::dt, ::ds, ::ds, :]

            self.times = X.shape[0]
            self.res = X.shape[1] 
            self.img_shape = X.shape[1:3] # (R,R,2)
            self.vars_shape_img = X.shape
            self.fields_shape_img = y.shape

            self.X = torch.from_numpy(X).float().view(-1,3)
            if add_vorticity:
                self.y = torch.from_numpy(y).float().view(-1,3)
            else:
                self.y = torch.from_numpy(y).float().view(-1,2)
            self.size = self.X.shape[0]

        assert self.X.shape[0] == self.y.shape[0]
                
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X = self.X[idx,:]
        y = self.y[idx,:]
        return X, y


class ResSynth2DDataset(Dataset):
    """
    Class to load Turb2D dataset in pytorch.
    The data follows the scikit-learn convenction: 
    
    if time is provided as array, it returns
    - X : (N, txy) -> (1000x256x256, 3)
    - y : (N, uv)  -> (1000x256x256, 2)

    if t is scalar, then it returns
    - X : (N, xy) -> (1000x256x256, 2)
    - u : (N, uv) -> (1000x256x256, 2)
    """

    def __init__(self,data_dir:str=None, ds:int=1, dt:int=1, time_idx:int=None, add_vorticity:bool=False):

        self.add_vorticity=add_vorticity
        
        dset = Synth2D(data_dir)
        dset.load_data(time_idx)
        self.diffusion = dset.diffusion

        self.ax_t = dset.t 
        self.ax_x = dset.x
        self.ax_y = dset.y

        # Data in Turb2D are (T,R,R,D)
        vars = dset.txy
        if add_vorticity:
            fields = dset.uvw
        else:
            fields = dset.uv

        nvars = vars.shape[-1]
        nfields = fields.shape[-1]
         
        assert nvars in [3]
        assert nfields in [2,3]

        vars = vars.squeeze()
        fields = fields.squeeze()

        if len(vars.shape) == 3: # single image (= one time shot)
            
            # downsampling
            X = vars[::ds, ::ds, 1:]
            y = fields[::ds, ::ds, :]

            print(X.shape)
            print(y.shape)
        
            self.res = X.shape[0] 

            self.img_shape = X.shape # (R,R,2)
            self.img_shape = X.shape[1:3] # (R,R,2)
            self.times = X.shape[0]
            self.res = X.shape[1] 
            self.vars_shape_img = X.shape
            self.fields_shape_img = y.shape

            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float()

            def resize(y, scale):
                if scale == 1:
                    return y
                res = y.shape[0]
                y = T.Resize(size=int(scale*res), antialias=True)(y.permute(2,0,1))
                return T.Resize(size=res)(y).permute(1,2,0)
            
            # low pass filter
            def lowpass_filter(y, k, s):
                print(f'Low passing with scale: {s}')
                # s = int(s/(2*3.14))
                # k = 6*s+1
                # return gaussian_blur2d(y.permute(2,0,1)[None,...], kernel_size = (k,k), sigma = (s,s))[0,...].permute(1,2,0)
                # return median_blur(y.permute(2,0,1)[None,...], [s,s])[0,...].permute(1,2,0)
                DFT = torch.fft.fft2(y.permute(2,0,1))
                DFT = torch.fft.fftshift(DFT)

                # plt.figure(figsize=(10,5))
                # plt.subplot(121)
                # plt.title(r'DFT of $u(x,y)$')
                # plt.imshow(torch.log(torch.abs(DFT)**2)[0,:,:])
                # plt.subplot(122)
                # plt.title(r'DFT of $v(x,y)$')
                # plt.imshow(torch.log(torch.abs(DFT)**2)[1,:,:])
                # plt.tight_layout()
                # plt.show()

                # Ideal Low Pass filter
                LP = torch.from_numpy(draw_circle(y.shape[:2], s))
                LP = LP / torch.max(torch.abs(LP))
                DFTlp = DFT * LP[None,:,:]
                ylp = (torch.fft.ifft2(torch.fft.ifftshift(DFTlp)).real).permute(1,2,0)
                
                # plt.figure(figsize=(15,5))
                # plt.suptitle('LP Filtering in DFT domain')
                # plt.subplot(131, title='Filter')
                # plt.imshow(LP.squeeze())
                # plt.subplot(132, title=r'LP of $U(k,l)$')
                # plt.imshow(torch.log(torch.abs(DFTlp[0,:,:])**2))
                # plt.subplot(133, title=r'LP of $V(k,l)$')
                # plt.imshow(torch.log(torch.abs(DFTlp[1,:,:])**2))
                # plt.tight_layout()
                # plt.show()


                # plt.figure(figsize=(10,5))
                # plt.subplot(121, title=r'Filtered $u(x,y)$')
                # plt.imshow(ylp[:,:,0])
                # plt.subplot(122, title=r'Filtered $v(x,y)$')
                # plt.imshow(ylp[:,:,1])
                # plt.tight_layout()
                # plt.show()
                return ylp


            def highpass_filter(y, k, s):
                ylp = lowpass_filter(y, k, s)
                return y - ylp

            # eps = 1e-5
            # y1f8 = torch.where(torch.abs(y1f8) < eps, eps*torch.ones_like(y1f8), y1f8)
            curr_max_res = y.shape[0]
            y1f8 = lowpass_filter(self.y.clone(), 11, curr_max_res//8)
            y1f4 = lowpass_filter(self.y.clone(), 11, curr_max_res//4)
            y1f2 = lowpass_filter(self.y.clone(), 11, curr_max_res//2)
            y1f1 = lowpass_filter(self.y.clone(), 11, curr_max_res//1)

            for y in [y1f8, y1f4, y1f2, y1f1]:
                plt.figure(figsize=(20,5))
                plt.subplot(141)
                plt.imshow(y[:,:,0])
                plt.colorbar()
                DFT = torch.fft.fft2(y[:,:,0])
                DFT = torch.fft.fftshift(DFT)
                plt.subplot(142)
                plt.imshow(torch.log(torch.abs(DFT)**2))
                plt.subplot(143)
                plt.imshow(y[:,:,1])
                plt.colorbar()
                DFT = torch.fft.fft2(y[:,:,1])
                DFT = torch.fft.fftshift(DFT)
                plt.subplot(144)
                plt.imshow(torch.log(torch.abs(DFT)**2))
                plt.tight_layout()
                plt.show()

            self.X = self.X.view(-1, 2)
            self.y = self.y.view(-1, 2)
            self.y1f8 = y1f8.view(-1, 2)
            self.y1f4 = y1f4.view(-1, 2)
            self.y1f2 = y1f2.view(-1, 2)
 
            assert self.y.shape == self.y1f2.shape == self.y1f4.shape == self.y1f8.shape
            self.size = self.X.shape[0]
        
        if len(vars.shape) == 4: # multiple images/times
            print('here')


            # remove boundaries
            C = 0
            if C > 0:
                vars = vars[:, C:-C, C:-C, :]
                fields = fields[:, C:-C, C:-C, :]

            # downsampling
            vars = vars[::dt, ::ds, ::ds, :]
            fields = fields[::dt, ::ds, ::ds, :]

            Rx = vars.shape[1]
            print(vars.shape)
            print(fields.shape)

            vars = torch.from_numpy(vars).float()
            fields = torch.from_numpy(fields).float()

            self.img_shape = vars.shape[1:3] # (R,R,2)
            self.times = vars.shape[0]
            self.res = vars.shape[1]
            self.vars_shape_img = vars.shape
            self.fields_shape_img = fields.shape

            # low pass fitering
            def loop_lowpass_fiter(y, s):
                for t in range(y.shape[0]):
                    y[t,...] = phy.lowpass_filter(y[t,...], s)

                    if t == 0:
                        
                        plt.subplot(131)
                        plt.title(s)
                        plt.imshow(y[t,:,:,0])
                        plt.subplot(132)
                        plt.imshow(y[t,:,:,1])
                        plt.subplot(133)
                        plt.imshow(y[t,:,:,2])
                        plt.show()

                return y

            self.y1f8 = loop_lowpass_fiter(fields.clone(), int(Rx / 8)).view(-1,3)
            self.y1f4 = loop_lowpass_fiter(fields.clone(), int(Rx / 4)).view(-1,3)
            self.y1f2 = loop_lowpass_fiter(fields.clone(), int(Rx / 2)).view(-1,3)
            self.y1f1 = loop_lowpass_fiter(fields.clone(), int(Rx / 1)).view(-1,3)

            self.X = vars.view(-1,3)
            if add_vorticity:
                self.y = fields.view(-1,3)
            else:
                self.y = fields.view(-1,2)

            self.size = self.X.shape[0]
        try:
            assert self.X.shape[0] == self.y.shape[0]
        except:
            print('X shape:', self.X.shape, 'vs y shape', self.y.shape)
                
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X = self.X[idx,:]
        y = self.y[idx,:]
        extra = (
            self.y1f8[idx,:],
            self.y1f4[idx,:],
            self.y1f2[idx,:],
        )

        return X, y, extra

class TurboFlowDataModule(pl.LightningDataModule):
    def __init__(self, dataset:str, data_dir:str, time_idx:int,
                 add_vorticity:bool,
                 train_batch_size:int, val_batch_size:int, test_batch_size:int,
                 train_downsampling_space:int, val_downsampling_space:int, test_downsampling_space:int,
                 train_downsampling_time:int,  val_downsampling_time:int,  test_downsampling_time:int,
                 train_shuffle:bool, val_shuffle:bool, test_shuffle:bool, num_workers:int):
        super().__init__()

        self.dataset = dataset

        if not self.dataset in currently_supported_dataset:
            raise ValueError(f'Supported Dataset are {currently_supported_dataset}, got: {dataset}')

        if self.dataset == 'Turb2D':
            self.dataset_fn = Turb2DDataset
        elif self.dataset == 'Re39000':
            self.dataset_fn = Re39000Dataset
        elif self.dataset == 'Synth2D':
            self.dataset_fn = ResSynth2DDataset
        else:
            raise ValueError('Provide Dataset name')

        self.data_dir = Path(data_dir)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.time_idx = time_idx
        self.add_vorticity = add_vorticity
        self.train_ds = train_downsampling_space
        self.val_ds = val_downsampling_space
        self.test_ds = test_downsampling_space
        self.train_dt = train_downsampling_time
        self.val_dt = val_downsampling_time
        self.test_dt = test_downsampling_time
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.num_workers = num_workers

    def print(self,):
        print('data_dir', self.data_dir)
        print('batch_size', self.batch_size)
        print('time_idx', self.time_idx)
        print('train_ds', self.train_ds)
        print('val_ds', self.val_ds)
        print('test_ds', self.test_ds)
        print('train_dt', self.train_dt)
        print('val_dt', self.val_dt)
        print('test_dt', self.test_dt)
        print('train_shuffle', self.train_shuffle)
        print('val_shuffle', self.val_shuffle)
        print('test_shuffle', self.test_shuffle)
        print('num_workers', self.num_workers)


    @staticmethod
    def add_data_specific_args(parent_parser):
        group = parent_parser.add_argument_group("data")
        group.add_argument("--dataset", type=str, required=True)
        group.add_argument("--data_dir", type=str, required=True)
        group.add_argument("--train_downsampling_space", type=int, default=4)
        group.add_argument("--val_downsampling_space", type=int, default=4)
        group.add_argument("--test_downsampling_space", type=int, default=1)
        group.add_argument("--train_downsampling_time", type=int, default=1)
        group.add_argument("--val_downsampling_time", type=int, default=1)
        group.add_argument("--test_downsampling_time", type=int, default=1)
        group.add_argument("--time_idx", type=int, default=42)
        group.add_argument("--train_batch_size", type=int, default=100000)
        group.add_argument("--val_batch_size", type=int, default=100000)
        group.add_argument("--test_batch_size", type=int, default=100000)
        group.add_argument("--add_vorticity", type=bool, default=False)
        group.add_argument("--train_shuffle", type=bool, default=False)
        group.add_argument("--val_shuffle", type=bool, default=False)
        group.add_argument("--test_shuffle", type=bool, default=False)
        group.add_argument("--num_workers", type=int, default=1)
        return parent_parser
        
    def prepare_data(self):
        # if download is required
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_fn(self.data_dir, self.train_ds, self.train_dt, self.time_idx, self.add_vorticity)
            self.val_dataset = self.dataset_fn(self.data_dir, self.val_ds, self.val_dt, self.time_idx, self.add_vorticity)

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_fn(self.data_dir, self.test_ds, self.test_dt, self.time_idx, self.add_vorticity)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.train_batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.val_batch_size, shuffle=self.val_shuffle, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.test_batch_size, shuffle=self.test_shuffle, num_workers=self.num_workers)


def load_turb2D_simple_numpy(path_to_turb2D:str='../data/2021-Turb2D_velocities.npy',ds:int=4,img:int=42):

    IMGs = np.load(path_to_turb2D)
    X = IMGs[img,::ds,::ds,:2] / 255
    y = IMGs[img,::ds,::ds,2:]

    # normalize output
    print('Y shape', y.shape)
    print('Y min, max:', np.min(y), np.max(y))
    y = y / np.max(np.abs(y))
    print('after normalization, Y min, max:', np.min(y), np.max(y))

    X = X.reshape(-1,2)
    y = y.reshape(-1,2)

    assert X.shape == y.shape

    return X, y


if __name__ == '__main__':
    pass