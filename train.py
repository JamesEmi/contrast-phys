import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import torch
from PhysNetModel import PhysNet
from loss import ContrastLoss
from IrrelevantPowerRatio import IrrelevantPowerRatio

from utils_data import *
from utils_sig import *
from torch import optim
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm
import matplotlib.pyplot as plt

ex = Experiment('model_train_trial', save_git_info=False)


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

@ex.config
def my_config():
    # here are some hyperparameters in our method

    # hyperparams for model training
    total_epoch = 30 # total number of epochs for training the model
    lr = 1e-5 # learning rate
    in_ch = 3 # TODO: number of input video channels, in_ch=3 for RGB videos, in_ch=1 for NIR videos.

    # hyperparams for ST-rPPG block
    fs = 30 # video frame rate, TODO: modify it if your video frame rate is not 30 fps.
    T = fs * 10 # temporal dimension of ST-rPPG block, default is 10 seconds.
    S = 2 # spatial dimenion of ST-rPPG block, default is 2x2.

    # hyperparams for rPPG spatiotemporal sampling
    delta_t = int(T/2) # time length of each rPPG sample
    K = 4 # the number of rPPG samples at each spatial position

    result_dir = './results' # store checkpoints and training recording
    ex.observers.append(FileStorageObserver(result_dir))


@ex.automain
def my_main(_run, total_epoch, T, S, lr, result_dir, fs, delta_t, K, in_ch):

    exp_dir = result_dir + '/%d'%(int(_run._id)) # store experiment recording to the path

    # get the training and test file path list by spliting the dataset
    train_list, test_list = UBFC_LU_split() # TODO: you should define your function to split your dataset for training and testing
    np.save(exp_dir+'/train_list.npy', train_list)
    np.save(exp_dir+'/test_list.npy', test_list)

    # define the dataloader
    train_dataset = H5Dataset(train_list, T) # please read the code about H5Dataset when preparing your dataset
    train_dataloader = DataLoader(train_dataset, batch_size=2, # two videos for contrastive learning
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True) # TODO: If you run the code on Windows, please remove num_workers=4.
    
    val_dataset = H5Dataset(test_list, T)  # Assuming validation dataset setup similarly
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # define the model and loss
    model = PhysNet(S, in_ch=in_ch).to(device).train()
    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250)

    # define irrelevant power ratio
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # define the optimizer
    opt = optim.AdamW(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    model.train()
    for e in range(total_epoch):
        with tqdm(total=int(np.round(60/(T/fs)).astype('int')), desc=f"Epoch {e+1}/{total_epoch}", unit='batch') as pbar:
            # for it in range(np.round(60/(T/fs)).astype('int')): # TODO: 60 means the video length of each video is 60s. If each video's length in your dataset is other value (e.g, 30s), you should use that value.
            for imgs in train_dataloader: # dataloader randomly samples a video clip with length T
                imgs = imgs.to(device)

                # model forward propagation
                model_output = model(imgs) 
                rppg = model_output[:,-1] # get rppg

                # define the loss functions
                loss, p_loss, n_loss = loss_func(model_output)

                train_losses.append(loss.item())
                # optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                # evaluate irrelevant power ratio during training
                ipr = torch.mean(IPR(rppg.clone().detach()))

                # save loss values and IPR
                ex.log_scalar("loss", loss.item())
                ex.log_scalar("p_loss", p_loss.item())
                ex.log_scalar("n_loss", n_loss.item())
                ex.log_scalar("ipr", ipr.item())

                pbar.update()  # Update the progress bar per iteration
        
        model.eval()
        with torch.no_grad():
            for imgs in val_dataloader:
                imgs = imgs.to(device)
                model_output = model(imgs)
                val_loss, _, _ = loss_func(model_output)
                val_losses.append(val_loss.item())

        torch.save(model.state_dict(), exp_dir + '/epoch%d.pt' % e)
        print(f'Epoch {e+1}: Train loss - {loss.item():.4f}; Val loss - {val_loss:.4f}')

    
    # Plotting losses after training
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
