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
from utils.dataset import PDEBenchDataset_Multi

# Import models
from models.LOD import LOD_Multi
from models.fno import FNO1d

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
    
    data_len = len(config.dataset.file_names)
    print("PDE parameter length:", data_len)

    model_name = config.model
    if model_name == 'fno':
        flag_POD = False
    else:
        flag_POD = True # True: POD 적용 모델


    # Initialize the dataset and dataloader
    train_data = PDEBenchDataset_Multi(file_names,
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                saved_folder=base_path,
                                flag_POD=flag_POD,
                                N_eigen=N_eigen)
    
    val_data = PDEBenchDataset_Multi(test_file_names,
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

    _, _data, _, _, _, bases, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    #print(bases.shape)
    print("Spatial Dimension", dimensions - 3)


    # Model define
    if model_name == 'fno':
        print("FNO 1D")
        model = FNO1d(num_channels=num_channels, 
                    modes=config.fno.modes, # 12, 
                    width=config.fno.width, # 20
                    initial_step=initial_step)

    elif model_name == 'lod':
        print("LOD")
        model = LOD_Multi(init_t=initial_step*num_channels,
                    N_eigen=N_eigen,
                    N_time=t_train*num_channels,
                    N_x=x_range,
                    bases=bases[0],
                    d_ff=x_range*2)
    
    else:
        raise Exception("There is no model.")

    os.makedirs('./checkpoint', exist_ok=True)
    model_path = "./checkpoint/" + model_name + "_" + config.dataset.name + '_multi.pt'
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

        for xx, yy, grid, pde_param, coeff, bases, bases_num in train_loader:
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

            if model_name == 'fno':
                for t in range(initial_step, t_train):
                        
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp = xx.reshape(inp_shape)
                    
                    # Extract target at current time step
                    y = yy[..., t:t+1, :]

                    # Model run
                    im = model(inp, grid) # (50, 128, 1, 1)

                    # Loss calculation
                    _batch = im.size(0)
                    loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
        
                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, im), -2)
        
                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy[..., :t_train, :]  # if t_train is not -1
                l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                train_l2_full += l2_full.item()
                train_l2.append(l2_full.item())

            #####
            elif model_name == 'lod':
                # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                inp = xx.reshape(inp_shape)

                y = yy[..., :t_train, :] # (N, S, T, 1)

                pred_pde, latent_pde, pred_coeff, latent_coeff = model(inp, bases_num)

                # Loss calculation
                latent_pde = latent_pde.permute(0,2,1)
                pred_pde = pred_pde.permute(0,2,1)
                _batch = pred_pde.size(0)

                # loss, loss_sub
                loss = loss_fn(pred_pde.reshape(_batch, -1), y.reshape(_batch, -1)) + loss_fn(latent_pde.reshape(_batch, -1), y.reshape(_batch, -1))
                #loss_sub = loss_fn(pred_coeff.reshape(_batch, -1), coeff.reshape(_batch, -1)) + loss_fn(latent_coeff.reshape(_batch, -1), coeff.reshape(_batch, -1))
                
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
                for xx, yy, grid, pde_param, coeff, bases, bases_num in val_loader:
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

                    if model_name == 'fno':
            
                        for t in range(initial_step, yy.shape[-2]):
                            inp = xx.reshape(inp_shape)
                            y = yy[..., t:t+1, :]

                            im = model(inp, grid)

                            _batch = im.size(0)
                            loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                            pred = torch.cat((pred, im), -2)
                
                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)
            
                        val_l2_step += loss.item()
                        _batch = yy.size(0)
                        _pred = pred[..., initial_step:t_train, :]
                        _yy = yy[..., initial_step:t_train, :]
                        l2_full = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                        val_l2_full += l2_full.item()
                        val_l2.append(l2_full.item())

                    elif  model_name == 'lod':
                        # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                        inp = xx.reshape(inp_shape)

                        # TODO: consider 2D
                        y = yy[..., initial_step:t_train, :] # (N, S, T, 1)

                        pred_pde, _, _, _ = model(inp, bases_num) # (N, T, seq)

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
    parser.add_argument('--pde', default='advection_multi')
    args = parser.parse_args()

    with open("./config/"+args.pde+'.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        config = Box(config)

    run_training(config)
    print("Done.")
