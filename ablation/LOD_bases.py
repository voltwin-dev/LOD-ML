import argparse
import yaml
from box import Box

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

# Import PDEBench dataloader
from utils.dataset import PDEBenchDataset

# Import models
from models.LOD_small import LOD_small, LOD_small_learnable

# Import function for counting model trainable parameters
from utils.utils import count_model_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(config):


    # Dataset setting
    test_path = base_path = "/data2/PDEBench/1D/"+config.dataset.name+"/Train/" # /data2/PDEBench/1D/Advection/Train/, /data2/PDEBench/1D/Burgers/Train/
    test_file_names = file_names = config.dataset.file_names
    t_train = config.dataset.t_train
    x_range = config.dataset.x_range
    initial_step = config.dataset.initial_step
    reduced_resolution = config.dataset.reduced_resolution
    reduced_resolution_t = config.dataset.reduced_resolution_t
    reduced_batch = config.dataset.reduced_batch
    num_channels = config.dataset.num_channels
    N_eigen = config.pod_parameter.N_eigen
    
    if config.dataset.name == 'ReactionDiffusion':
        parameter = config.dataset.file_names[0].split("_")[1] + "_" + config.dataset.file_names[0].split("_")[-1][:-5]
    else:
        parameter = config.dataset.file_names[0].split("_")[-1][:-5]
    print("PDE parameter:", parameter)

    model_name = config.model
    flag_POD = True # True: POD 적용 모델


    # Initialize the dataset and dataloader
    train_data = PDEBenchDataset(file_names,
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder=base_path,
                                flag_POD=flag_POD,
                                N_eigen=N_eigen)
    
    val_data = PDEBenchDataset(test_file_names,
                            reduced_resolution=reduced_resolution,
                            reduced_resolution_t=reduced_resolution_t,
                            reduced_batch=reduced_batch,
                            initial_step=initial_step,
                            if_test=True,
                            saved_folder=test_path,
                            flag_POD=flag_POD,
                            N_eigen=N_eigen)


    # Hyperparmaeters setting
    num_workers = config.training.num_workers
    model_update = 1
    batch_size = config.training.batch_size
    epochs = config.training.epochs
    learning_rate = config.training.learning_rate
    random_seed = config.training.random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


    # Define dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    _, _data, _, _, _, bases = next(iter(val_loader))
    dimensions = len(_data.shape)
    #print(bases.shape)
    print("Spatial Dimension", dimensions - 3)


    # Model define
    if model_name == 'lod-small':
        print("LOD_small")
        model = LOD_small(init_t=initial_step,
                        N_eigen=N_eigen,
                        N_time=t_train,
                        N_x=x_range)

    elif model_name == 'lod-small-learnable':
        model = LOD_small_learnable(init_t=initial_step, # POD_Upgrade
                            N_eigen=N_eigen,
                            N_time=t_train,
                            N_x=x_range,
                            bases=bases[0])
    
    else:
        raise Exception("There is no model.")

    os.makedirs('./checkpoint', exist_ok=True)
    model_path = "./checkpoint/abl_bases_" + model_name + "_" + config.dataset.name + "_" + parameter + '.pt'
    print("Model name:", model_path)

    model.to(device)
    total_params = count_model_params(model)
    print(f"Total Trainable Parameters = {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    loss_fn = nn.MSELoss()
    loss_val_min = np.infty


    # Training
    for ep in tqdm(range(epochs)):
        print("### Epoch: ", ep, "###")
        model.train()
        train_l2_step = 0
        train_l2_full = 0
        train_l2 = []

        for xx, yy, grid, pde_param, coeff, bases in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
            pde_param = pde_param.to(device)
            bases = bases[0].to(device)
            coeff = coeff.to(device)

            pred = yy[..., :initial_step, :]
            #print(pred.shape)

            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)

            if model_name == 'lod-small':

                inp = xx.reshape(inp_shape)

                y = yy[..., :t_train, :] # (N, S, T, 1)

                # Model run
                pred_coeff = model(inp) # (N, T, Eigen)

                # Loss calculation
                im = torch.einsum('btn, tns -> bts', pred_coeff, bases).permute(0, 2, 1) # (N, S, T)
                _batch = im.size(0)
                loss = loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1)) #+ loss_fn(pred_coeff, coeff.to(device))

                train_l2_full += loss.item()
                train_l2.append(loss.item())

            #####
            elif model_name == 'lod-small-learnable':
                # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                inp = xx.reshape(inp_shape)

                y = yy[..., :t_train, :] # (N, S, T, 1)

                pred_pde, pred_coeff = model(inp)

                # Loss calculation
                pred_pde = pred_pde.permute(0,2,1)
                _batch = pred_pde.size(0)

                # loss, loss_sub
                loss = loss_fn(pred_pde.reshape(_batch, -1), y.reshape(_batch, -1))
                
                train_l2_full += loss.item()
                train_l2.append(loss.item())

            # Optimize
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        if ep % model_update == 0:
            val_l2_step = 0
            val_l2_full = 0
            val_l2 = []
            model.eval()
            print("### Testing... ###")

            with torch.no_grad():
                for xx, yy, grid, pde_param, coeff, bases in val_loader:
                    loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)
                    pde_param = pde_param.to(device)
                    bases = bases[0].to(device)

                    pred = yy[..., :initial_step, :]
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    if model_name == 'lod-small':
                        # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                        inp = xx.reshape(inp_shape)

                        # TODO: consider 2D
                        y = yy[..., initial_step:t_train, :] # (N, S, T, 1)

                        pred_coeff = model(inp) # (N, T, Eigen)

                        # Loss calculation
                        im = torch.einsum('btn, tns -> bts', pred_coeff, bases).permute(0, 2, 1) # (N, S, T)
                        im = im[:, :, initial_step:]
                        _batch = im.size(0)
                        loss = loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                        val_l2_full += loss.item()
                        val_l2.append(loss.item())

                    elif  model_name == 'lod-small-learnable':
                        # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                        inp = xx.reshape(inp_shape)

                        # TODO: consider 2D
                        y = yy[..., initial_step:t_train, :] # (N, S, T, 1)

                        pred_pde, _ = model(inp) # (N, T, seq)

                        # Loss calculation
                        pred_pde = pred_pde.permute(0,2,1)
                        im = pred_pde[..., initial_step:]
                        _batch = im.size(0)
                        loss = loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                        val_l2_full += loss.item()
                        val_l2.append(loss.item())
                
                val_l2 = np.average(val_l2)
                train_l2 = np.average(train_l2)

                # Save checkpoint
                if val_l2_full < loss_val_min:
                    print("### Model Saving...###")
                    loss_val_min = val_l2_full
                    torch.save({
                        "epoch": ep,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        #"scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss_val_min
                    }, model_path)
            model.train()

        scheduler.step()
        print('epoch: {0}, trainMSE: {1:.7f} | testMSE: {2:.7f} | testAVG_MSE: {3:.7f} | testAVG_RMSE: {4:.7f}'\
                .format(ep, train_l2, val_l2_full, val_l2, val_l2**(1/2)))
        

# Run script
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='config argparser')
    parser.add_argument('--pde', default='abl_bases')
    args = parser.parse_args()

    with open("./config/"+args.pde+'.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        config = Box(config)

    run_training(config)
    print("Done.")
