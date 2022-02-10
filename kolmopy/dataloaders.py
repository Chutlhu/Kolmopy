from datetime import time
import glob
import numpy as np
from pathlib import Path
from natsort import natsorted

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from kolmopy.datasets.turb2D import Turb2D
from kolmopy.core import Dataset

currently_supported_dataset = ['Turb2D', 'Re39000']

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
            self.img_shape = X.shape[1:3] # (R,R,2)

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


class TurboFlowDataModule(pl.LightningDataModule):
    def __init__(self, dataset:str, data_dir:str, batch_size:int, time_idx:int,
                 train_downsampling_space:int, val_downsampling_space:int, test_downsampling_space:int,
                 train_downsampling_time:int,  val_downsampling_time:int,  test_downsampling_time:int,
                 train_shuffle:bool, val_shuffle:bool, test_shuffle:bool, num_workers:int):
        super(TurboFlowDataModule, self).__init__()

        self.dataset = dataset

        if not self.dataset in currently_supported_dataset:
            raise ValueError(f'Supported Dataset are {currently_supported_dataset}, got: {dataset}')

        if self.dataset == 'Turb2D':
            self.dataset_fn = Turb2DDataset
        if self.dataset == 'Re39000':
            self.dataset_fn = Re39000Dataset

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.time_idx = time_idx
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
        group.add_argument("--batch_size", type=int, default=100000)
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
            self.train_dataset = self.dataset_fn(self.data_dir, self.train_ds, self.train_dt, self.time_idx)
            self.val_dataset = self.dataset_fn(self.data_dir, self.val_ds, self.val_dt, self.time_idx)

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_fn(self.data_dir, self.test_ds, self.train_dt, self.time_idx)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=self.val_shuffle, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=self.test_shuffle, num_workers=self.num_workers)


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