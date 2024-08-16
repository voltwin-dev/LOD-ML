import os
import re
import math as mt

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm import tqdm

class PDEBenchDataset(Dataset):
    """
    Loads data in PDEBench format. Slightly adaped code from PDEBench.
    """

    def __init__(self, filenames,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 truncated_trajectory_length=-1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1,
                 use_save_file=False,
                 flag_POD=False,
                 N_eigen=32):
        """
        Represent dataset that consists of PDE with different parameters.

        :param filenames: filenames that contain the datasets
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional
        :param truncated_trajectory_length: cuts temporal subsampled trajectory yielding a trajectory of given length. -1 means that trajectory is not truncated
        :type truncated_trajectory_length: INT, optional

        """

        # Also accept single file name
        if type(filenames) == str:
            filenames = [filenames]

        self.pod_path = '/data2/PDEBench/POD/'
        self.data = np.array([])
        self.pde_parameter = np.array([])

        # Load data
        def load(filename, num_samples_max, test_ratio):
            root_path = os.path.abspath(saved_folder + filename)
            print("### Loading...", root_path)
            #assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'

            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()

                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    if len(idx_cfd)==3:  # 1D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              3],
                                             dtype=np.float32)
                        #density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,2] = _data   # batch, x, t, ch

                        grid = np.array(f["x-coordinate"], dtype=np.float32)
                        grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                        print(data.shape)
                    if len(idx_cfd)==4:  # 2D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,3] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

                    if len(idx_cfd)==5:  # 3D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              idx_cfd[4]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              5],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,3] = _data   # batch, x, t, ch
                        # Vz
                        _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z)
                        grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution, \
                                    ::reduced_resolution, \
                                    ::reduced_resolution]

                else:  # scalar equations
                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                    if len(_data.shape) == 3:  # 1D
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data = _data[:, :, :, None]  # batch, x, t, ch

                        grid = np.array(f["x-coordinate"], dtype=np.float32)
                        grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        data = np.concatenate([_data, data], axis=-1)
                        data = data[:, :, :, :, None]  # batch, x, y, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, data.shape[0])
            else:
                num_samples_max = data.shape[0]

            test_idx = int(num_samples_max * test_ratio)

            if if_test:
                data = data[-test_idx:]
            else:
                data = data[:-test_idx]

            # Get pde parameter from file name
            matches = re.findall(r"_[a-zA-Z]+([0-9].[0-9]+|1.e-?[0-9]+)", filename)
            pde_parameter_scalar = [float(match) for match in matches]
            pde_parameter = np.tile(pde_parameter_scalar, (data.shape[0], 1)).astype(np.float32)

            return data, pde_parameter, grid

        data, pde_parameter, grid = zip(*Parallel(n_jobs=len(filenames))(delayed(load)(filename, num_samples_max, test_ratio) for filename in filenames))
        self.data = np.vstack(data)
        self.pde_parameter = np.vstack(pde_parameter)
        self.grid = grid[0]

        #print(self.data.shape) # (9000, 256, 41, 1)
        t_range = self.data.shape[-2]
        x_range = self.data.shape[1]

        _num = self.data.shape[0]
        
        if flag_POD:

            pde_name = self.pod_path + filenames[0][:-5] + '_pde.npy'
            coeff_name = self.pod_path + filenames[0][:-5] + '_coeff.npy'
            bases_name = self.pod_path + filenames[0][:-5] + '_bases.npy'

            if not if_test:
                self.data = np.load(pde_name)[:_num, ...] # 9000
                if len(self.data.shape) == 3:
                    self.data = self.data.transpose(0, 2, 1)[..., None] # Reshape No!!!! (permute or transpose)
                    #print(self.data.shape) # (9000, 256, 41, 1)
                else:
                    pass
                self.coeff = np.load(coeff_name) # 9000개
                self.bases = np.load(bases_name)

            else:
                if len(self.data.shape) == 3:
                    print("Test set")
                    self.data = np.load(pde_name)[-_num:, ...] # -1000
                    self.data = self.data.transpose(0, 2, 1)[..., None] # Reshape No!!!! (permute or transpose)
                    #print(self.data.shape) # (9000, 256, 41, 1)
                else:
                    pass
                print(self.data.shape)
                self.coeff = np.zeros((1000, t_range, N_eigen))
                self.bases = np.load(bases_name)

            print("Coeff shape:", self.coeff.shape)
            print("Bases shape:", self.bases.shape)

        else:
            self.coeff = np.zeros((9000, t_range, N_eigen))
            self.bases = np.zeros((t_range, N_eigen, x_range))

        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.data = torch.tensor(self.data)
        self.pde_parameter = torch.tensor(self.pde_parameter)

        # truncate trajectory
        if truncated_trajectory_length > 0:
            self.data = self.data[..., :truncated_trajectory_length, :]        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        xx = self.data[idx, ..., :self.initial_step, :].type(torch.FloatTensor)
        yy = self.data[idx].type(torch.FloatTensor)
        grid = self.grid.type(torch.FloatTensor)
        parameter = self.pde_parameter[idx].type(torch.FloatTensor)
        coeff = torch.FloatTensor(self.coeff[idx])
        bases = torch.FloatTensor(self.bases)

        return xx, yy, grid, parameter, coeff, bases

class PDEBenchDataset_Multi(Dataset):
    """
    Loads data in PDEBench format. Slightly adaped code from PDEBench.
    """

    def __init__(self, filenames,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 truncated_trajectory_length=-1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1,
                 use_save_file=False,
                 flag_POD=False,
                 N_eigen=32):
        """
        Represent dataset that consists of PDE with different parameters.

        :param filenames: filenames that contain the datasets
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional
        :param truncated_trajectory_length: cuts temporal subsampled trajectory yielding a trajectory of given length. -1 means that trajectory is not truncated
        :type truncated_trajectory_length: INT, optional

        """

        # Also accept single file name
        if type(filenames) == str:
            filenames = [filenames]

        self.pod_path = '/data2/PDEBench/POD/'
        self.data = np.array([])
        self.pde_parameter = np.array([])
        self.data_len = len(filenames)

        # Load data
        def load(filename, num_samples_max, test_ratio):
            root_path = os.path.abspath(saved_folder + filename)
            print("### Loading...", root_path)
            #assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'

            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()

                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    if len(idx_cfd)==3:  # 1D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              3],
                                             dtype=np.float32)
                        #density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,2] = _data   # batch, x, t, ch

                        grid = np.array(f["x-coordinate"], dtype=np.float32)
                        grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                        print(data.shape)
                    if len(idx_cfd)==4:  # 2D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,3] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

                    if len(idx_cfd)==5:  # 3D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              idx_cfd[4]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              5],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,3] = _data   # batch, x, t, ch
                        # Vz
                        _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z)
                        grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution, \
                                    ::reduced_resolution, \
                                    ::reduced_resolution]

                else:  # scalar equations
                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                    if len(_data.shape) == 3:  # 1D
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data = _data[:, :, :, None]  # batch, x, t, ch

                        grid = np.array(f["x-coordinate"], dtype=np.float32)
                        grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        data = np.concatenate([_data, data], axis=-1)
                        data = data[:, :, :, :, None]  # batch, x, y, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, data.shape[0])
            else:
                num_samples_max = data.shape[0]

            test_idx = int(num_samples_max * test_ratio)

            if if_test:
                data = data[:test_idx]
            else:
                data = data[test_idx:num_samples_max]

            # Get pde parameter from file name
            matches = re.findall(r"_[a-zA-Z]+([0-9].[0-9]+|1.e-?[0-9]+)", filename)
            pde_parameter_scalar = [float(match) for match in matches]
            pde_parameter = np.tile(pde_parameter_scalar, (data.shape[0], 1)).astype(np.float32)

            return data, pde_parameter, grid

        # n_jobs = len(filenames)
        data, pde_parameter, grid = zip(*Parallel(n_jobs=4)(delayed(load)(filenames[i], num_samples_max, test_ratio) for i in range(len(filenames))))
        self.data = np.vstack(data)
        self.pde_parameter = np.vstack(pde_parameter)
        #print(self.data.shape, self.pde_parameter.shape)
        #print(grid[0]==grid[-1])
        #print(grid[0].shape)
        #self.grid = np.concatenate([grid[0], grid[-1]], axis=-1) # (Seq, 2)
        self.grid = grid[0]

        #print(self.data.shape) # (9000, 256, 41, 1)
        t_range = self.data.shape[-2]
        x_range = self.data.shape[1]

        _num = self.data.shape[0]
        self._num = _num//len(filenames)

        ## 7/4: just 1 equation only 
        #### after developing for multi equation
        if flag_POD:

            self.data = np.zeros((_num, x_range, t_range, 1))
            self.coeff = np.zeros((_num, t_range, N_eigen))
            self.bases = np.zeros((self.data_len, t_range, N_eigen, x_range))

            each_num = _num // self.data_len

            for j in range(self.data_len):

                pde_name = self.pod_path + filenames[j][:-5] + '_pde.npy'
                coeff_name = self.pod_path + filenames[j][:-5] + '_coeff.npy'
                bases_name = self.pod_path + filenames[j][:-5] + '_bases.npy'

                #pde_name = self.pod_path + 'Advection_beta4_pde.npy'
                #coeff_name = self.pod_path + 'Advection_beta4_pod_coeff.npy'
                #bases_name = self.pod_path + 'Advection_beta4_pod_bases.npy'

                if not if_test:

                    data = np.load(pde_name)[:9000, ...]
                    self.data[each_num*j:each_num*(j+1)] = data.transpose(0, 2, 1)[..., None] # Reshape No!!!! (permute or transpose)
                    #print(self.data.shape) # (9000, 256, 41, 1)
                    self.coeff[each_num*j:each_num*(j+1)] = np.load(coeff_name) # 9000개
                    self.bases[j] = np.load(bases_name)

                else:
                    data = np.load(pde_name)[9000:, ...]
                    self.data[each_num*j:each_num*(j+1)] = data.transpose(0, 2, 1)[..., None] # (1000, 256, 41, 1)
                    print(self.data.shape)
                    self.coeff[each_num*j:each_num*(j+1)] = np.zeros((1000, t_range, N_eigen))
                    self.bases[j] = np.load(bases_name)

            print("Coeff shape:", self.coeff.shape)
            print("Bases shape:", self.bases.shape)

        else:
            self.coeff = np.zeros((9000*self.data_len, t_range, N_eigen))
            self.bases = np.zeros((t_range, N_eigen, x_range))

        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.data = torch.tensor(self.data)
        self.pde_parameter = torch.tensor(self.pde_parameter)

        # truncate trajectory
        if truncated_trajectory_length > 0:
            self.data = self.data[..., :truncated_trajectory_length, :]        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        `self.data` is already subsampled across time and space.
        `self.grid` is already subsampled
        """
        xx = self.data[idx, ..., :self.initial_step, :].type(torch.FloatTensor)
        yy = self.data[idx].type(torch.FloatTensor)
        parameter = self.pde_parameter[idx].type(torch.FloatTensor)
        coeff = torch.FloatTensor(self.coeff[idx])
        bases = torch.FloatTensor(self.bases)

        bases_num = idx//self._num
        #grid = torch.FloatTensor(self.grid[..., bases_num])[..., None]
        grid = self.grid.type(torch.FloatTensor)
        #print(grid.shape) #(seq, 1)

        #return self.data[idx, ..., :self.initial_step, :], self.data[idx], self.grid, self.pde_parameter[idx], self.coeff[idx], self.bases
        return xx, yy, grid, parameter, coeff, bases, bases_num
    
class PDEBenchDataset_Datasize(Dataset):
    """
    Loads data in PDEBench format. Slightly adaped code from PDEBench.
    """

    def __init__(self, filenames,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 truncated_trajectory_length=-1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1,
                 data_num = 10000,
                 use_save_file=False,
                 flag_POD=False,
                 N_eigen=32):
        """
        Represent dataset that consists of PDE with different parameters.

        :param filenames: filenames that contain the datasets
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional
        :param truncated_trajectory_length: cuts temporal subsampled trajectory yielding a trajectory of given length. -1 means that trajectory is not truncated
        :type truncated_trajectory_length: INT, optional
        """

        # Also accept single file name
        if type(filenames) == str:
            filenames = [filenames]

        self.pod_path = '/data2/PDEBench/POD/'
        self.data = np.array([])
        self.pde_parameter = np.array([])

        self.test_num = 1000
        self.train_num = data_num - self.test_num

        # Load data
        def load(filename, num_samples_max, test_ratio):
            root_path = os.path.abspath(saved_folder + filename)
            print("### Loading...", root_path)
            #assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'

            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()

                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    if len(idx_cfd)==3:  # 1D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              3],
                                             dtype=np.float32)
                        #density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,2] = _data   # batch, x, t, ch

                        grid = np.array(f["x-coordinate"], dtype=np.float32)
                        grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                        print(data.shape)
                    if len(idx_cfd)==4:  # 2D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,3] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

                    if len(idx_cfd)==5:  # 3D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              idx_cfd[4]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              5],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,3] = _data   # batch, x, t, ch
                        # Vz
                        _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z)
                        grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution, \
                                    ::reduced_resolution, \
                                    ::reduced_resolution]

                else:  # scalar equations
                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                    if len(_data.shape) == 3:  # 1D
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data = _data[:, :, :, None]  # batch, x, t, ch

                        grid = np.array(f["x-coordinate"], dtype=np.float32)
                        grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        data = np.concatenate([_data, data], axis=-1)
                        data = data[:, :, :, :, None]  # batch, x, y, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, data.shape[0])
            else:
                num_samples_max = data.shape[0]

            test_idx = int(num_samples_max * test_ratio)

            if if_test:
                data = data[data_num-self.test_num:data_num]
            else:
                data = data[:data_num-self.test_num]

            # Get pde parameter from file name
            matches = re.findall(r"_[a-zA-Z]+([0-9].[0-9]+|1.e-?[0-9]+)", filename)
            pde_parameter_scalar = [float(match) for match in matches]
            pde_parameter = np.tile(pde_parameter_scalar, (data.shape[0], 1)).astype(np.float32)

            return data, pde_parameter, grid

        data, pde_parameter, grid = zip(*Parallel(n_jobs=len(filenames))(delayed(load)(filename, num_samples_max, test_ratio) for filename in filenames))
        self.data = np.vstack(data)
        self.pde_parameter = np.vstack(pde_parameter)
        self.grid = grid[0]

        #print(self.data.shape) # (9000, 256, 41, 1)
        t_range = self.data.shape[-2]
        x_range = self.data.shape[1]

        _num = self.data.shape[0]

        ## 7/4: just 1 equation only 
        #### after developing for multi equation =
        if flag_POD:

            pde_name = self.pod_path + filenames[0][:-5] + '_' + str(data_num) + '_pde.npy'
            coeff_name = self.pod_path + filenames[0][:-5] + '_' + str(data_num) + '_coeff'+str(N_eigen)+'.npy'
            bases_name = self.pod_path + filenames[0][:-5] + '_' + str(data_num) + '_bases'+str(N_eigen)+'.npy'

            if not if_test:
                self.data = np.load(pde_name)[:_num, ...]
                if len(self.data.shape) == 3:
                    self.data = self.data.transpose(0, 2, 1)[..., None] # Reshape No!!!! (permute or transpose)
                    #print(self.data.shape) # (9000, 256, 41, 1)
                else:
                    pass # CFD (N, 128, 21 ,3)
                self.coeff = np.load(coeff_name) # 9000개
                self.bases = np.load(bases_name)

            else:
                if len(self.data.shape) == 3:
                    self.data = np.load(pde_name)[-_num:, ...]
                    self.data = self.data.transpose(0, 2, 1)[..., None] # Reshape No!!!! (permute or transpose)
                    #print(self.data.shape) # (9000, 256, 41, 1)
                else:
                    pass # CFD (N, 128, 21 ,3)
                print(self.data.shape)
                self.coeff = np.zeros((1000, t_range, N_eigen))
                self.bases = np.load(bases_name)

            print("Coeff shape:", self.coeff.shape)
            print("Bases shape:", self.bases.shape)

        else:
            self.coeff = np.zeros((self.train_num, t_range, N_eigen))
            self.bases = np.zeros((t_range, N_eigen, x_range))

        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.data = torch.tensor(self.data)
        self.pde_parameter = torch.tensor(self.pde_parameter)

        # truncate trajectory
        if truncated_trajectory_length > 0:
            self.data = self.data[..., :truncated_trajectory_length, :]        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        `self.data` is already subsampled across time and space.
        `self.grid` is already subsampled
        """
        xx = self.data[idx, ..., :self.initial_step, :].type(torch.FloatTensor)
        yy = self.data[idx].type(torch.FloatTensor)
        grid = self.grid.type(torch.FloatTensor)
        parameter = self.pde_parameter[idx].type(torch.FloatTensor)
        coeff = torch.FloatTensor(self.coeff[idx])
        bases = torch.FloatTensor(self.bases)

        #return self.data[idx, ..., :self.initial_step, :], self.data[idx], self.grid, self.pde_parameter[idx], self.coeff[idx], self.bases
        return xx, yy, grid, parameter, coeff, bases
    
class PDEBenchDataset_Eigenvalues(Dataset):
    """
    Loads data in PDEBench format. Slightly adaped code from PDEBench.
    """

    def __init__(self, filenames,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 truncated_trajectory_length=-1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1,
                 data_num = 10000,
                 use_save_file=False,
                 flag_POD=False,
                 N_eigen=32):
        """
        Represent dataset that consists of PDE with different parameters.

        :param filenames: filenames that contain the datasets
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional
        :param truncated_trajectory_length: cuts temporal subsampled trajectory yielding a trajectory of given length. -1 means that trajectory is not truncated
        :type truncated_trajectory_length: INT, optional
        """

        # Also accept single file name
        if type(filenames) == str:
            filenames = [filenames]

        self.pod_path = '/data2/PDEBench/POD/'
        self.data = np.array([])
        self.pde_parameter = np.array([])

        self.test_num = 1000
        self.train_num = data_num - self.test_num

        # Load data
        def load(filename, num_samples_max, test_ratio):
            root_path = os.path.abspath(saved_folder + filename)
            print("### Loading...", root_path)
            #assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'

            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()

                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    if len(idx_cfd)==3:  # 1D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              3],
                                             dtype=np.float32)
                        #density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data[...,2] = _data   # batch, x, t, ch

                        grid = np.array(f["x-coordinate"], dtype=np.float32)
                        grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                        print(data.shape)
                    if len(idx_cfd)==4:  # 2D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        data[...,3] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

                    if len(idx_cfd)==5:  # 3D
                        data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              idx_cfd[4]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              5],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,3] = _data   # batch, x, t, ch
                        # Vz
                        _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z)
                        grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution, \
                                    ::reduced_resolution, \
                                    ::reduced_resolution]

                else:  # scalar equations
                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                    if len(_data.shape) == 3:  # 1D
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        data = _data[:, :, :, None]  # batch, x, t, ch

                        grid = np.array(f["x-coordinate"], dtype=np.float32)
                        grid = torch.tensor(grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        data = np.concatenate([_data, data], axis=-1)
                        data = data[:, :, :, :, None]  # batch, x, y, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, data.shape[0])
            else:
                num_samples_max = data.shape[0]

            test_idx = int(num_samples_max * test_ratio)

            if if_test:
                data = data[data_num-self.test_num:data_num]
            else:
                data = data[:data_num-self.test_num]

            # Get pde parameter from file name
            matches = re.findall(r"_[a-zA-Z]+([0-9].[0-9]+|1.e-?[0-9]+)", filename)
            pde_parameter_scalar = [float(match) for match in matches]
            pde_parameter = np.tile(pde_parameter_scalar, (data.shape[0], 1)).astype(np.float32)

            return data, pde_parameter, grid

        data, pde_parameter, grid = zip(*Parallel(n_jobs=len(filenames))(delayed(load)(filename, num_samples_max, test_ratio) for filename in filenames))
        self.data = np.vstack(data)
        self.pde_parameter = np.vstack(pde_parameter)
        self.grid = grid[0]

        #print(self.data.shape) # (9000, 256, 41, 1)
        t_range = self.data.shape[-2]
        x_range = self.data.shape[1]

        _num = self.data.shape[0]

        ## 7/4: just 1 equation only 
        #### after developing for multi equation =
        if flag_POD:

            pde_name = self.pod_path + filenames[0][:-5] + '_' + str(data_num) + '_pde.npy'
            coeff_name = self.pod_path + filenames[0][:-5] + '_' + str(data_num) + '_coeff'+str(N_eigen)+'.npy'
            bases_name = self.pod_path + filenames[0][:-5] + '_' + str(data_num) + '_bases'+str(N_eigen)+'.npy'

            if not if_test:
                self.data = np.load(pde_name)[:_num, ...]
                if len(self.data.shape) == 3:
                    self.data = self.data.transpose(0, 2, 1)[..., None] # Reshape No!!!! (permute or transpose)
                    #print(self.data.shape) # (9000, 256, 41, 1)
                else:
                    pass # CFD (N, 128, 21 ,3)
                self.coeff = np.load(coeff_name) # 9000개
                self.bases = np.load(bases_name)

            else:
                if len(self.data.shape) == 3:
                    self.data = np.load(pde_name)[-_num:, ...]
                    self.data = self.data.transpose(0, 2, 1)[..., None] # Reshape No!!!! (permute or transpose)
                    #print(self.data.shape) # (9000, 256, 41, 1)
                else:
                    pass # CFD (N, 128, 21 ,3)
                print(self.data.shape)
                self.coeff = np.zeros((1000, t_range, N_eigen))
                self.bases = np.load(bases_name)

            print("Coeff shape:", self.coeff.shape)
            print("Bases shape:", self.bases.shape)

        else:
            self.coeff = np.zeros((self.train_num, t_range, N_eigen))
            self.bases = np.zeros((t_range, N_eigen, x_range))

        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.data = torch.tensor(self.data)
        self.pde_parameter = torch.tensor(self.pde_parameter)

        # truncate trajectory
        if truncated_trajectory_length > 0:
            self.data = self.data[..., :truncated_trajectory_length, :]        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        `self.data` is already subsampled across time and space.
        `self.grid` is already subsampled
        """
        xx = self.data[idx, ..., :self.initial_step, :].type(torch.FloatTensor)
        yy = self.data[idx].type(torch.FloatTensor)
        grid = self.grid.type(torch.FloatTensor)
        parameter = self.pde_parameter[idx].type(torch.FloatTensor)
        coeff = torch.FloatTensor(self.coeff[idx])
        bases = torch.FloatTensor(self.bases)

        #return self.data[idx, ..., :self.initial_step, :], self.data[idx], self.grid, self.pde_parameter[idx], self.coeff[idx], self.bases
        return xx, yy, grid, parameter, coeff, bases