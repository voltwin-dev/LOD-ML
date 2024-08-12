import argparse
import yaml
from box import Box

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
            assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'

            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()

                if 'tensor' not in keys:
                    _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    print("Dataset num:", idx_cfd[0])
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

            #if if_test:
            #    data = data[:test_idx]
            #else:
            #    data = data[test_idx:num_samples_max]

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
        #t_range = self.data.shape[-2]
        #x_range = self.data.shape[1]

        #_num = self.data.shape[0]

        ## 7/4: just 1 equation only 
        #### after developing for multi equation =

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
        xx = self.data[idx]

        #return self.data[idx, ..., :self.initial_step, :], self.data[idx], self.grid, self.pde_parameter[idx], self.coeff[idx], self.bases
        return xx

# function define
def POD_reshape(T):
    ns = T.shape[0]
    print("Original shape of T:", T.shape)  # 원본 T의 형태 출력
    T = T.reshape(ns, -1)
    print("Reshaped T:", T.shape)  # 변환된 T의 형태 출력

    return T

def POD(T, N_eigen):
    
    #print(T.shape)

  # Eigenvalue problem
    U = T @ T.T
    #print("Reshaped U:", U.shape)
    if (U==U.T).all(): # symmetric
      D, V = np.linalg.eigh(U)
    else:
      print('Not symmetric')
      D, V = np.linalg.eig(U)
    #print("Reshaped V:", V.shape)
    #print("Reshaped D:", D.shape)

    del U
    
    # Sorting eigenvalues and eigenvectors
    indices = D.argsort()[::-1]
    D = D[indices]
    V = V[:, indices]
    
    # Calculating cumulative energy ratio
    cumulative_energy_ratio = np.cumsum(D) / np.sum(D)
    #print(cumulative_energy_ratio >= 1 - epsilon)
    
    # Finding the number of eigenvalues to satisfy the energy threshold
    # n = np.argmax(cumulative_energy_ratio >= 1 - epsilon) + 1 # False/True로 표현되었으므로...
    n = N_eigen
    #print("(Default) Number of eigenvalues to retain:", n)
    
    # Normalizing eigenvectors
    EV = V[:, :n] / np.sqrt(D[:n])
    #print("Reshaped EV:", EV.shape)
    
    # Calculating the projection matrix
    phi = EV.T @ T
    #print("Reshaped phi:", phi.shape)
    
    # Reconstructing T
    Tr = T @ phi.T
    #print("Reshaped Tr:", Tr.shape)

    return Tr, phi, cumulative_energy_ratio

if __name__ == '__main__':

    ####################### Dataset name listup ###########################
    # 1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5
    # 1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5
    ######################################################################

    # Load config
    print("## Loading config...")
    parser = argparse.ArgumentParser(description='config argparser')
    parser.add_argument('--pde', default='make_1D_CFD_POD')
    args = parser.parse_args()

    with open("./config/"+args.pde+'.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        config = Box(config)

    ############# 우리가 활용할 데이터셋
    data_path = config.dataset.data_path
    print("Data:", data_path[0])
    #############

    ## Default values
    t_range = 21 # 101
    x_range = 128 # 1024
    initial_step = 10
    reduced_resolution = 8 # 4
    reduced_resolution_t = 5 # 5
    reduced_batch = 1 # 10000 -> 1000
    n_channels = 3
    file_name = data_path[0][:-5]

    print("T range:", t_range)
    print("X range:", x_range)

    # Variables setting
    N_eigen = config.dataset.N_eigen
    print("## POD making ##\n", "N_eigen:", N_eigen)

    root_path = config.dataset.root_path
    save_path = config.dataset.save_path
    pde_save_path = save_path+file_name+'_pde.npy'
    coeff_save_path = save_path+file_name+'_coeff.npy'
    bases_save_path = save_path+file_name+'_bases.npy'

    # PDE Dataset
    pde_dataset = PDEBenchDataset(data_path,
                            reduced_resolution=reduced_resolution,
                            reduced_resolution_t=reduced_resolution_t,
                            reduced_batch=reduced_batch,
                            initial_step=initial_step,
                            saved_folder=root_path,
                            use_save_file=False,
                            flag_POD=False)

    train_pde_dataloader = torch.utils.data.DataLoader(pde_dataset, batch_size=1, shuffle=False) # 16

    # PDE stacking
    print("Stacking...")
    pde_data = np.zeros((10000, x_range, t_range, n_channels))
    for i, (xx) in enumerate(train_pde_dataloader):
        pde_data[i] = xx

    print("## PDE Saving ##")
    np.save(pde_save_path, pde_data) # save

    # POD 수행하는 코드
    pde_data = np.load('/data2/PDEBench/POD/'+file_name+'_pde.npy') # save

    pde_data = pde_data.transpose(0,2,1,3)
    print("pde_Data shape:", pde_data.shape) # (10000, 21, 128, 3)

    density_pde = pde_data[..., 0]
    pressure_pde = pde_data[..., 1]
    velocity_pde = pde_data[..., 2]

    all_coeff_data = np.zeros((3, 9000, t_range, N_eigen))
    all_bases_data = np.zeros((3, t_range, N_eigen, 128))

    ## density
    coeff_data = np.zeros((9000, t_range, N_eigen))
    bases_data = np.zeros((t_range, N_eigen, x_range))

    for t in tqdm(range(t_range)):

        t_input = density_pde[:9000, t, :]
            
        coeff, basis, _ = POD(t_input, N_eigen, t_range)
        coeff_data[:, t, :] = coeff[:9000, :]
        bases_data[t] = basis

    all_coeff_data[0] = coeff_data
    all_bases_data[0] = bases_data

    ## Pressure
    coeff_data = np.zeros((9000, t_range, N_eigen))
    bases_data = np.zeros((t_range, N_eigen, x_range))
    for t in tqdm(range(t_range)):
        #if t < 10:
        #    t_input = pde_data[:10000, t, :]
        #else:
        t_input = pressure_pde[:9000, t, :]
            
        coeff, basis, _ = POD(t_input, N_eigen, t_range)
        coeff_data[:, t, :] = coeff[:9000, :]
        bases_data[t] = basis

    all_coeff_data[1] = coeff_data
    all_bases_data[1] = bases_data

    ## density
    coeff_data = np.zeros((9000, t_range, N_eigen))
    bases_data = np.zeros((t_range, N_eigen, x_range))
    for t in tqdm(range(t_range)):
        #if t < 10:
        #    t_input = pde_data[:10000, t, :]
        #else:
        t_input = velocity_pde[:9000, t, :]
            
        coeff, basis, _ = POD_2D(t_input, N_eigen, t_range)
        coeff_data[:, t, :] = coeff[:9000, :]
        bases_data[t] = basis

    all_coeff_data[2] = coeff_data
    all_bases_data[2] = bases_data


    print("## Coeff & Bases Saving ##")
    np.save('/data2/PDEBench/POD/'+file_name+'_coeff.npy', all_coeff_data) # (4, 900, t, N_eigen)
    np.save('/data2/PDEBench/POD/'+file_name+'_bases.npy', all_bases_data) # (4, t, N_eigen, xyc)
    # ----------------------------------------------------------------------------- 여기까지 수행하면, POD 데이터 생성 완료

    # POD Error
    for i in range(3):
        coeff_t = all_coeff_data[i]
        #print(coeff_t.shape)
        basis_t = all_bases_data[i]
        #print(basis_t.shape)

        pod_pde = np.einsum('btn, tns -> bts', coeff_t, basis_t)
        pod_pde = pod_pde.transpose(0,2,1)
        print(pod_pde.shape) # (t, seq) = (t_range, x_range)

        rmse_value = np.mean(((pod_pde - pde_data[:9000, ..., 2])**2))**(1/2) # RMSE: 6.090103551002303e-06
        print("POD RMSE:", rmse_value)

    print("## END Prorcess ##")