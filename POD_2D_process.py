##### Module Import
import argparse
import yaml
from box import Box

import random
import pandas as pd
import torch
import numpy as np
import h5py
from tqdm import tqdm

#### Class define
import os
import re
import math as mt

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm import tqdm

class PDEBenchDataset_Water(Dataset):
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
                #print(keys)

                data_list = []
                for i in f:
                    data = f[i]['data']
                    grid = np.array(f[i]['grid']['x'], dtype=np.float32)
                    data_list.append(np.array(data)[np.newaxis, ...])
                shallow_water = np.concatenate(data_list, axis=0) # (N, t, r, 1)
                shallow_water = shallow_water[..., 0]
                data = np.array(shallow_water)

            #print(data.shape) # (1000, 101, 128, 128)
            data = data.transpose(0, 2, 3, 1)[..., None]
            grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)

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
        print("self.data shape:", self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        `self.data` is already subsampled across time and space.
        `self.grid` is already subsampled
        """
        data = self.data[idx]

        #return self.data[idx, ..., :self.initial_step, :], self.data[idx], self.grid, self.pde_parameter[idx], self.coeff[idx], self.bases
        return data
    
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

    # Load config
    print("## Loading config...")
    parser = argparse.ArgumentParser(description='config argparser')
    parser.add_argument('--pde', default='make_2D_POD')
    args = parser.parse_args()

    with open("./config/"+args.pde+'.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        config = Box(config)

    ############# 우리가 활용할 데이터셋
    data_path = config.dataset.data_path
    print("Data:", data_path[0])
    #############

    ## Default values
    t_range = 101 # 101
    x_range = 128 # 1024
    initial_step = 10
    reduced_resolution = 1 # 4
    reduced_resolution_t = 1 # 5
    reduced_batch = 1
    n_channels = 1
    file_name = data_path[0][:-3]

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
    pde_dataset = PDEBenchDataset_Water(data_path,
                            reduced_resolution=reduced_resolution,
                            reduced_resolution_t=reduced_resolution_t,
                            reduced_batch=reduced_batch,
                            initial_step=initial_step,
                            saved_folder=root_path,
                            use_save_file=False,
                            flag_POD=False)

    train_pde_dataloader = torch.utils.data.DataLoader(pde_dataset, batch_size=1, shuffle=False) # 16

    # PDE stacking
    pde_data = np.zeros((1000, x_range, x_range, t_range, 1))
    for i, (batch) in enumerate(train_pde_dataloader):
        pde_data[i] = batch

    print("## PDE Saving ##")
    np.save(pde_save_path, pde_data) # save

    # POD 수행하는 코드
    x_range = 128*128 # change

    coeff_data = np.zeros((900, t_range, N_eigen))
    bases_data = np.zeros((t_range, N_eigen, x_range))
    for t in tqdm(range(t_range)):

        t_input = pde_data[:900, ..., t, 0]
        t_input = t_input.reshape(900, -1)
        #print(t_input.shape) # (900, 16384)
            
        coeff, basis, _ = POD(t_input, N_eigen)
        coeff_data[:, t, :] = coeff[:900, :]
        bases_data[t] = basis


    print("## Coeff & Bases Saving ##")
    np.save(coeff_save_path, coeff_data) # save
    np.save(bases_save_path, bases_data) # save
    # ----------------------------------------------------------------------------- 여기까지 수행하면, POD 데이터 생성 완료

    print("## PDE Reconstruction error ##")
    coeff_data = np.load(coeff_save_path) # save
    bases_data = np.load(bases_save_path) # save
    pde_data = np.load(pde_save_path) # save


    coeff_t = coeff_data
    print("coeff shape:", coeff_t.shape)
    basis_t = bases_data
    print("bases shape:", basis_t.shape)

    pod_pde = np.einsum('btn, tns -> bts', coeff_t, basis_t)
    pod_pde = pod_pde.reshape(900, 101, 128, 128).transpose(0, 2, 3, 1)
    print("Reconstruction shape:", pod_pde.shape)

    rmse_value = np.mean(((pod_pde - pde_data[:900])**2))**(1/2) # RMSE: 6.090103551002303e-06
    print("POD RMSE:", rmse_value)

    print("## END Prorcess ##")