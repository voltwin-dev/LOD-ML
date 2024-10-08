import argparse
import yaml
from box import Box

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

# Import PDEBench dataloader
from utils.dataset import PDEBenchDataset_water

# Import models
from models.LOD import LOD_2D
from models.LOD_small import LOD_small
from models.fno import FNO2d

# Import function for counting model trainable parameters
from utils.utils import count_model_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(config):


    # Dataset setting
    if config.dataset.name == 'shallow-water':
        test_path = base_path = "/data2/PDEBench/2D/"+config.dataset.name+"/"
    else:
        raise Exception("No implementation")
    test_file_names = file_names = config.dataset.file_names
    t_train = config.dataset.t_train
    x_range = config.dataset.x_range
    initial_step = config.dataset.initial_step
    reduced_resolution = config.dataset.reduced_resolution
    reduced_resolution_t = config.dataset.reduced_resolution_t
    reduced_batch = config.dataset.reduced_batch
    num_channels = config.dataset.num_channels
    N_eigen = config.pod_parameter.N_eigen
    
    if config.dataset.name == 'shallow-water':
        parameter = "NA"
    else:
        raise Exception("No implementation")
    print("PDE parameter:", parameter)

    model_name = config.model
    if model_name == 'fno':
        flag_POD = False
    else:
        flag_POD = True # True: POD 적용 모델


    # Initialize the dataset and dataloader
    if config.dataset.name == 'shallow-water':
        val_data = PDEBenchDataset_water(test_file_names,
                                reduced_resolution=reduced_resolution,
                                reduced_resolution_t=reduced_resolution_t,
                                reduced_batch=reduced_batch,
                                initial_step=initial_step,
                                if_test=True,
                                saved_folder=test_path,
                                flag_POD=flag_POD,
                                N_eigen=N_eigen)
    else:
        raise Exception("No implementation")

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
        print("FNO 2D")
        model = FNO2d(num_channels=num_channels, 
                        modes1=config.fno.modes, # 12, 
                        modes2=config.fno.modes, # 12, 
                        width=config.fno.width, # 20, 64
                        initial_step=initial_step)
            
    elif model_name == 'lod-small':
        print("LOD_small")
        model = LOD_small(init_t=initial_step,
                        N_eigen=N_eigen,
                        N_time=t_train,
                        N_x=x_range**2)

    elif model_name == 'lod':
        print("LOD")
        model = LOD_2D(init_t=initial_step*num_channels,
                        N_eigen=N_eigen,
                        N_time=t_train*num_channels,
                        N_x=x_range**2,
                        bases=bases[0],
                        )
    
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

    pred_pde = np.zeros((1000, x_range, x_range, t_train, channel))
    gt_pde = np.zeros((1000, x_range, x_range, t_train, channel))

    # Training
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

                _pred = pred[..., initial_step:t_train, :]
                _yy = yy[..., initial_step:t_train, :]
                loss = loss_fn(_pred, _yy)
                mse_loss.append(loss.item())

                pred_pde[i] = pred.cpu()

            elif model_name == 'lod-small':

                inp = xx.reshape(-1, x_range**2, initial_step)

                y = yy[..., initial_step:t_train, :] # (N, S, T, 1)

                pred_coeff = model(inp) # (N, T, Eigen)

                # Loss calculation
                im = torch.einsum('btn, tns -> bts', pred_coeff, bases).reshape(-1, t_train, x_range, x_range) # (N, S, T)
                im = im.permute(0, 2, 3, 1) # (N, S, T, C)
                _pred = im[..., initial_step:]
                _batch = _pred.size(0)

                loss = loss_fn(_pred.reshape(_batch, -1), y.reshape(_batch, -1))

                pred_pde[i] = im[..., None].cpu()

                mse_loss.append(loss.item())

            elif  model_name == 'lod':
                inp = xx.reshape(-1, x_range**2, initial_step)

                y = yy[..., initial_step:t_train, :] # (N, S, T, 1)

                prediction_pde, _, _, _ = model(inp) # (N, T, seq)
                prediction_pde = prediction_pde.reshape(-1, t_train, x_range, x_range)
                prediction_pde = prediction_pde.permute(0, 2, 3, 1)
                im = prediction_pde[..., initial_step:]

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
    print("### Save 99th data...")
    k=99 # 189, 235, 226
    c = 0
    pred = pred_pde[k, ..., c]
    gt = gt_pde[k, ..., c]

    ims = []
    fig, ax = plt.subplots()
    #plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #plt.margins(0, 0)
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())

    for i in tqdm(range(pred.shape[-1])):
        line1 = ax.imshow(pred[..., i].squeeze(), animated=True)  # prediction
        #line2 = ax.imshow(gt[..., i].squeeze(), animated=True)  # ground-trurh

        ims.append([line1])
        #ims.append([line2])

    # Animate the plot
    os.makedirs('./visualization', exist_ok=True)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save('./visualization/'+str(k)+'.gif', writer=writer)
    print("### Animation saved ###")
            

# Run script
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='config argparser')
    parser.add_argument('--pde', default='shallow-water')
    args = parser.parse_args()

    with open("./config/"+args.pde+'.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        config = Box(config)

    run_training(config)
    print("Done.")
