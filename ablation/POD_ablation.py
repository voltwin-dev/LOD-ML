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
class CustomDataset:
    def __init__(self, 
                 path, 
                 data_dim=['1D'],
                 data_name=[''],
                 mix=False,
                 reduced_resolution=4,
                 reduced_resolution_t=5,
                 reduced_batch=1,
                 limit_dataset=10000):
        '''
        Description (in paper)
        1. 1-Dimension PDEBench dataset (t=T; predicted)
        - Advection(201) (t=15)
        - Burgers(201) (t=15)
        - CFD(101) (t=5)
        - ReactionDiffusion(201) (t=15)
        - diffusion-sorption(101) (t=15)

        2. 2-Dimension PDEBench dataset
        - (Coming soon)

        3. 3-Dimension PDEBench dataset
        - (Coming soon)
        '''

        # Define default variables
        self.path = path
        self.data_name = data_name
        self.train_data = []
        self.train_name = []
        self.reduced_resolution=reduced_resolution
        self.reduced_resolution_t=reduced_resolution_t
        self.reduced_batch=reduced_batch
        self.limit_num = limit_dataset
        print("### Warning: We recommend -> # of dataset", limit_dataset, "in H100 GPU. ###")

        # mix or not mix
        ## mix: 1D, 2D + 3D, ...
        ## not mix: Choose one
        if not mix:
            self.data_dim = data_dim[0] # Assign first element

            if self.data_dim == '1D':
                self.load_1D_dataset()

            elif self.data_dim == '2D':
                self.load_2D_dataset()

            elif self.data_dim == '3D':
                self.load_3D_dataset()

        else:
            # 아직 1D_dimension 밖에 고려 안 됨.
            self.data_dim = data_dim
            
            if '1D' in self.data_dim: # TODO: Not Yet
                self.load_1D_dataset()

            if '2D' in self.data_dim: # TODO: Not Yet
                self.load_2D_dataset()

            if '3D' in self.data_dim: # TODO: Not Yet
                self.load_3D_dataset()
    
    # Advection_load function
    def advection_load(self, filename, length):

        if self.data_dim == '1D':
            data_path = self.path + '/Advection/Train/' + filename
            
            # loading
            with h5py.File(data_path, 'r') as f:
                advection_1d = f['tensor'] # (10000, 201, 1024)
                self.adv_data = np.array(advection_1d)[:,::self.reduced_resolution_t,::self.reduced_resolution]
            self.train_data.append(self.adv_data[:length])
            print(self.adv_data.shape)

        else:
            raise Exception('Not yet')
        
    # Burgers_load function
    def burgers_load(self, filename, length):

        if self.data_dim == '1D':
            data_path = self.path + '/Burgers/Train/' + filename
            
            # loading
            with h5py.File(data_path, 'r') as f:
                burgers_1d = f['tensor'] # (10000, 201, 1024)
                self.burg_data = np.array(burgers_1d)[:,::self.reduced_resolution_t,::self.reduced_resolution]
            self.train_data.append(self.burg_data[:length])
            print(self.burg_data.shape)

        else:
            raise Exception('Not yet')
        
    # CFD_load function
    def CFD_load(self, filename, length):

        if self.data_dim == '1D':
            data_path = self.path + '/CFD/Train/' + filename
            
            # loading
            with h5py.File(data_path, 'r') as f:
                CFD_1d = f['Vx'] # (10000, 101, 1024) // # TODO: Vx means? (논의...)
                self.CFD_data = np.array(CFD_1d)[:,:,::self.reduced_resolution]
            self.train_data.append(self.CFD_data[:length])
            print(self.CFD_data.shape)

        else:
            raise Exception('Not yet')
    
    # reactiondiffusion_load function
    def reactiondiffusion_load(self, filename, length):

        if self.data_dim == '1D':
            data_path = self.path + '/ReactionDiffusion/Train/' + filename
            
            # loading
            with h5py.File(data_path, 'r') as f:
                react_diffusion_1d = f['tensor'] # (10000, 201, 1024) // 101?
                self.react_diff_data = np.array(react_diffusion_1d)[:,:,::self.reduced_resolution]
            self.train_data.append(self.react_diff_data[:length])
            print(self.react_diff_data.shape)

        else:
            raise Exception('Not yet')
    
    # diffusion_sorption_load function
    ## 1D-mension에서 해당 데이터만 101이므로, reduced_resolution_t 적용 x 
    def diffusion_sorption_load(self, filename, length):

        if self.data_dim == '1D':
            data_path = self.path + '/diffusion-sorption/' + filename
            
            # loading
            with h5py.File(data_path, 'r') as f:
                data_list = []
                for i in f:
                    data = f[i]['data']
                    data_list.append(np.array(data)[np.newaxis, ...])
                diffusion_sorp_1d = np.concatenate(data_list, axis=0) # (N, t, r, 1)
                diffusion_sorp_1d = diffusion_sorp_1d[..., 0]
                self.diffusion_sorp_data = np.array(diffusion_sorp_1d)[:,:,::self.reduced_resolution]
            self.train_data.append(self.diffusion_sorp_data[:length])
            print(self.diffusion_sorp_data.shape)

        else:
            raise Exception('Not yet')
        
    def make_onehot(self):
        t = torch.arange(0, self.data_name_length)
        a = torch.zeros((self.data_name_length, self.data_name_length))
        a[range(self.data_name_length), t] = 1
        self.data_cls_onehot = a
        
    # load 1D dataset
    ## TODO: 변수가 1024개 있다고 보는 것이 맞는지? (Yes)
    def load_1D_dataset(self):

        # self.path = /data2/PDEBench/1D
        print('Using datasets:', len(self.data_name))
        self.data_name_length = len(self.data_name)
        self.data_each_length = self.limit_num // self.data_name_length

        for name in tqdm(self.data_name):
            
            # Data loading...
            if 'Advection' in name:
                self.advection_load(name, self.data_each_length)
                name_list = ['Advection']*self.data_each_length
                self.train_name += name_list

            elif 'Burgers' in name:
                self.burgers_load(name, self.data_each_length)
                name_list = ['Burgers']*self.data_each_length
                self.train_name += name_list

            elif 'ReacDiff' in name:
                self.reactiondiffusion_load(name, self.data_each_length)
                name_list = ['ReacDiff']*self.data_each_length
                self.train_name += name_list

            elif 'diff-sorp' in name:
                self.diffusion_sorption_load(name, self.data_each_length)
                name_list = ['diff-sorp']*self.data_each_length
                self.train_name += name_list

            elif 'CFD' in name:
                self.CFD_load(name, self.data_each_length)
                name_list = ['CFD']*self.data_each_length
                self.train_name += name_list

            else:
                raise Exception('Unknown dataset or uncorrect path.')

        self.each_data_len = [len(d) for d in self.train_data]
        self.make_onehot()

    # load 2D dataset
    def load_2D_dataset(self):

        # self.path = /data2/PDEBench/1D
        print('Using datasets:', len(self.data_name))
        self.data_name_length = len(self.data_name)

        raise Exception('아직 구현 안 함.') 
    
    # load 3D dataset
    def load_3D_dataset(self):

        # self.path = /data2/PDEBench/1D
        print('Using datasets:', len(self.data_name))
        self.data_name_length = len(self.data_name)
        
        raise Exception('아직 구현 안 함.') 

    def __len__(self):
        count = 0
        for data in self.train_data:
            count += len(data)
        return count

    def __getitem__(self, idx):
        '''
        pde_data: Sequence_data.
        self.data_name[random_num]: For generating prompt.
        '''
        # data number
        random_num = random.choice(range(self.data_name_length))
        # idx in each dataset
        # idx = random.choice(range(0, self.data_each_length))
        
        pde_data = torch.FloatTensor(self.train_data[random_num][idx]) # If contain many datasets, select one type
        return pde_data, self.data_cls_onehot[random_num], self.data_name[random_num]
    
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
    # Advection path
    advection_path = ['1D_Advection_Sols_beta0.1.hdf5', '1D_Advection_Sols_beta0.2.hdf5', \
                    '1D_Advection_Sols_beta0.4.hdf5', '1D_Advection_Sols_beta0.7.hdf5', \
                    '1D_Advection_Sols_beta1.0.hdf5', '1D_Advection_Sols_beta2.0.hdf5', \
                    '1D_Advection_Sols_beta4.0.hdf5', '1D_Advection_Sols_beta7.0.hdf5']

    # Burgers path
    burgers_path = ['1D_Burgers_Sols_Nu0.001.hdf5', '1D_Burgers_Sols_Nu0.002.hdf5', '1D_Burgers_Sols_Nu0.004.hdf5', \
                    '1D_Burgers_Sols_Nu0.01.hdf5', '1D_Burgers_Sols_Nu0.02.hdf5', '1D_Burgers_Sols_Nu0.04.hdf5', \
                    '1D_Burgers_Sols_Nu0.1.hdf5', '1D_Burgers_Sols_Nu0.2.hdf5', '1D_Burgers_Sols_Nu0.4.hdf5', \
                    '1D_Burgers_Sols_Nu1.0.hdf5', '1D_Burgers_Sols_Nu2.0.hdf5', '1D_Burgers_Sols_Nu4.0.hdf5']

    # ReactionDiff path
    reactdiff_path = ['ReacDiff_Nu0.5_Rho1.0.hdf5', 'ReacDiff_Nu0.5_Rho10.0.hdf5', 'ReacDiff_Nu0.5_Rho2.0.hdf5',
                    'ReacDiff_Nu0.5_Rho5.0.hdf5', 'ReacDiff_Nu1.0_Rho1.0.hdf5', 'ReacDiff_Nu1.0_Rho10.0.hdf5',
                    'ReacDiff_Nu1.0_Rho2.0.hdf5', 'ReacDiff_Nu1.0_Rho5.0.hdf5', 'ReacDiff_Nu2.0_Rho1.0.hdf5',
                    'ReacDiff_Nu2.0_Rho10.0.hdf5', 'ReacDiff_Nu2.0_Rho2.0.hdf5', 'ReacDiff_Nu2.0_Rho5.0.hdf5',
                    'ReacDiff_Nu5.0_Rho1.0.hdf5', 'ReacDiff_Nu5.0_Rho10.0.hdf5', 'ReacDiff_Nu5.0_Rho2.0.hdf5',
                    'ReacDiff_Nu5.0_Rho5.0.hdf5'] 

    # SorpDiff
    sorpdiff_path = ['1D_diff-sorp_NA_NA.h5'] # 1개
    ######################################################################

    # Load config
    print("## Loading config...")
    parser = argparse.ArgumentParser(description='config argparser')
    parser.add_argument('--pde', default='ablation_POD')
    args = parser.parse_args()

    with open("./config/"+args.pde+'.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        config = Box(config)

    ############# 우리가 활용할 데이터셋
    data_path = config.dataset.data_path
    print("Data:", data_path[0])
    #############

    ## Default values
    if 'ReacDiff' in data_path[0]:
        t_range = 101 # 41 or 201
        x_range = 256 # 256 or 1024
        initial_step = 5 # 5 (Reactdiff)
        reduced_resolution = 4 # 4
        reduced_resolution_t = 1 # 5
        reduced_batch = 1
        file_name = data_path[0][:-5]
        print(file_name)

    elif 'diff-sorp' in data_path[0]:
        t_range = 101 # 41 or 201
        x_range = 1024 # 256 or 1024
        initial_step = 10 # 5 (Reactdiff)
        reduced_resolution = 1 # 4
        reduced_resolution_t = 1 # 5
        reduced_batch = 1
        file_name = data_path[0][:-3] # .h5
        print(file_name)

    else: # Advection, Burgers
        t_range = 41 # 41 or 201
        x_range = 256 # 256 or 1024
        initial_step = 10 # 5 (Reactdiff)
        reduced_resolution = 4 # 4
        reduced_resolution_t = 5 # 5
        reduced_batch = 1
        file_name = data_path[0][:-5]
        print(file_name)

    print("T range:", t_range)
    print("X range:", x_range)

    # Variables setting
    N_eigen = config.dataset.N_eigen
    print("## POD making ##\n", "N_eigen:", N_eigen)
    limit_dataset_num = config.dataset.data_num # 10000, 5500, 4000, 2500
    print("Limit dataset:", limit_dataset_num)
    limit_train_num = limit_dataset_num - 1000
    print("Train dataset:", limit_train_num)

    root_path = config.dataset.root_path
    save_path = config.dataset.save_path
    pde_save_path = save_path+file_name+ '_' + str(limit_dataset_num) + '_pde.npy'
    coeff_save_path = save_path+file_name+ '_' + str(limit_dataset_num) + '_coeff' + str(N_eigen) + '.npy'
    bases_save_path = save_path+file_name+ '_' + str(limit_dataset_num) + '_bases' + str(N_eigen) + '.npy'

    # PDE Dataset
    pde_dataset = CustomDataset(root_path, 
                                data_dim=['1D'], 
                                data_name=data_path, 
                                mix=False, 
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                limit_dataset=limit_dataset_num)

    train_pde_dataloader = torch.utils.data.DataLoader(pde_dataset, batch_size=1, shuffle=False) # 16

    # PDE stacking
    pde_data = np.zeros((limit_dataset_num, t_range, x_range))
    for i, (batch, batch_cls, batch_name) in enumerate(train_pde_dataloader):
        pde_data[i] = batch

    print("## PDE Saving ##")
    np.save(pde_save_path, pde_data) # save

    # POD 수행하는 코드
    coeff_data = np.zeros((limit_train_num, t_range, N_eigen))
    bases_data = np.zeros((t_range, N_eigen, x_range))
    for t in tqdm(range(t_range)):
        #if t < 10:
        #    t_input = pde_data[:10000, t, :]
        #else:
        t_input = pde_data[:limit_train_num, t, :]
            
        coeff, basis, _ = POD(t_input, N_eigen)
        coeff_data[:, t, :] = coeff[:limit_train_num, :]
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
    print("Reconstruction shape:", pod_pde.shape)

    rmse_value = np.mean(((pod_pde - pde_data[:limit_train_num])**2))**(1/2) # RMSE: 6.090103551002303e-06
    print("POD RMSE:", rmse_value)

    print("## END Prorcess ##")