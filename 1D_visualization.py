import argparse
import yaml
from box import Box

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

# Import PDEBench dataloader
from utils.dataset import PDEBenchDataset, PDEBenchDataset_Sorp

# Import models
from models.LOD import LOD, LOD_Sorp
from models.LOD_small import LOD_small
from models.fno import FNO1d

# Import function for counting model trainable parameters
from utils.utils import count_model_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(config):


    # Dataset setting
    if config.dataset.name == 'diffusion-sorption':
        test_path = base_path = "/data2/PDEBench/1D/"+config.dataset.name+"/"
    else:
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
    elif config.dataset.name == 'diffusion-sorption':
        parameter = "NA"
    else:
        parameter = config.dataset.file_names[0].split("_")[-1][:-5]
    print("PDE parameter:", parameter)

    model_name = config.model
    if model_name == 'fno':
        flag_POD = False
    else:
        flag_POD = True # True: POD 적용 모델


    # Initialize the dataset and dataloader
    if config.dataset.name == 'diffusion-sorption':

        val_data = PDEBenchDataset_Sorp(test_file_names,
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                if_test=True,
                                saved_folder=test_path,
                                flag_POD=flag_POD,
                                N_eigen=N_eigen)

    else:
        
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
    val_loader = DataLoader(val_data, batch_size=1, num_workers=num_workers, shuffle=False)

    _, _data, _, _, _, bases = next(iter(val_loader))
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
            
    elif model_name == 'lod-small':
        print("LOD_small")
        model = LOD_small(init_t=initial_step,
                        N_eigen=N_eigen,
                        N_time=t_train,
                        N_x=x_range)

    elif model_name == 'lod':
        print("LOD")
        if config.dataset.name == 'diffusion-sorption':
            model = LOD_Sorp(init_t=initial_step*num_channels,
                            N_eigen=N_eigen,
                            N_time=t_train*num_channels,
                            N_x=x_range,
                            bases=bases[0],
                            d_ff=x_range*2)
        else:
            model = LOD(init_t=initial_step*num_channels,
                            N_eigen=N_eigen,
                            N_time=t_train*num_channels,
                            N_x=x_range,
                            bases=bases[0],
                            d_ff=x_range*2)
    
    else:
        raise Exception("There is no model.")

    os.makedirs('./checkpoint', exist_ok=True)
    model_path = "./checkpoint/" + model_name + "_" + config.dataset.name + "_" + parameter + '.pt'
    print("Model name:", model_path)

    model.to(device)
    total_params = count_model_params(model)
    print(f"Total Trainable Parameters = {total_params}")

    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()

    model.to(device)

    # need something
    loss_fn = nn.MSELoss()
    channel = 1
    mse_loss = []

    pred_pde = np.zeros((1000, x_range, t_train, channel))
    gt_pde = np.zeros((1000, x_range, t_train, channel))

    # inference
    with torch.no_grad():
        for i, (xx, yy, grid, pde_param, coeff, bases) in tqdm(enumerate(val_loader)):
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

            gt_pde[i] = yy.cpu()

            if model_name == 'fno':

                for t in range(initial_step, yy.shape[-2]):
                    inp = xx.reshape(inp_shape)
                    y = yy[..., t:t+1, :]

                    im = model(inp, grid)

                    _batch = im.size(0)
                    loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                    pred = torch.cat((pred, im), -2)
        
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                # VRAM
                if i == 0:
                    def to_MB(a):
                        return a/1024.0/1024.0
                        
                    print(f"After model to device: {to_MB(torch.cuda.memory_allocated()):.2f}MB")

                _pred = pred[..., initial_step:t_train, :]
                _yy = yy[..., initial_step:t_train, :]
                loss = loss_fn(_pred, _yy)
                mse_loss.append(loss.item())

                pred_pde[i] = pred.cpu()
                
            elif model_name == 'lod-small':
                # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                inp = xx.reshape(inp_shape)
                y = yy[..., initial_step:t_train, :] # (N, S, T, 1)

                pred_coeff = model(inp) # (N, T, Eigen)
                im = torch.einsum('btn, tns -> bts', pred_coeff, bases).permute(0, 2, 1) # (N, S, T)

                # VRAM
                if i == 0:
                    def to_MB(a):
                        return a/1024.0/1024.0
                        
                    print(f"After model to device: {to_MB(torch.cuda.memory_allocated()):.2f}MB")

                _pred = im[:, :, initial_step:]
                _batch = im.size(0)
                loss = loss_fn(_pred.reshape(_batch, -1), y.reshape(_batch, -1))

                pred_pde[i] = im[..., None].cpu()

                mse_loss.append(loss.item())

            elif  model_name == 'lod':
                # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                inp = xx.reshape(inp_shape)
                y = yy[..., initial_step:t_train, :] # (N, S, T, 1)

                prediction_pde, _, _, _ = model(inp) # (N, T, seq)

                # VRAM
                if i == 0:
                    def to_MB(a):
                        return a/1024.0/1024.0
                        
                    print(f"After model to device: {to_MB(torch.cuda.memory_allocated()):.2f}MB")

                prediction_pde = prediction_pde.permute(0,2,1)
                im = prediction_pde[:, :, initial_step:]

                _batch = im.size(0)
                loss = loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                pred_pde[i] = prediction_pde[..., None].cpu()
                mse_loss.append(loss.item())

    # Evaluation
    mse = np.average(mse_loss)
    rmse = mse**(1/2)
    print("MSE:", round(mse,7))
    print("RMSE:", round(rmse,7))

    # Visualization
    print("### Save 999th data...")
    k = 999
    c = 0
    pred = pred_pde[k, ..., c]
    gt = gt_pde[k, ..., c]

    ims = []
    fig, ax = plt.subplots()
    for i in tqdm(range(pred.shape[1])):
        line1 = ax.plot(pred[:, i].squeeze(), animated=True, color="blue", label='fno')  # prediction
        line2 = ax.plot(gt[:, i].squeeze(), animated=True, color="red", label='gt')  # ground-trurh

        ax.plot
        #ax.set_xlabel('Spatial')
        #ax.set_ylabel('tensor(y)')
        #ax.set_title(str(k)+'; Init seq: [0, 10)')
        #ax.legend(loc='upper center')
        ims.append([line1[0], line2[0]])

    # Animate the plot
    os.makedirs('./visualization', exist_ok=True)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save('./visualization/'+str(k)+'.gif', writer=writer)
    print("### Animation saved ###")
            

# Run script
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='config argparser')
    parser.add_argument('--pde', default='advection')
    args = parser.parse_args()

    with open("./config/"+args.pde+'.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        config = Box(config)

    run_training(config)
    print("Done.")
