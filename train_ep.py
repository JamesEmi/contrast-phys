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
from datetime import datetime
from torch.optim.lr_scheduler import StepLR


ex = Experiment('model_train_equipleth_finetune', save_git_info=False)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

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

    result_dir = './results/equipleth' # store checkpoints and training recording
    ex.observers.append(FileStorageObserver(result_dir))


@ex.automain
def my_main(_run, total_epoch, T, S, lr, result_dir, fs, delta_t, K, in_ch):

    exp_dir = result_dir + '/%d'%(int(_run._id)) # store experiment recording to the path

    # get the training and test file path list by spliting the dataset
    # train_list, test_list = SQH_split() # TODO: you should define your function to split your dataset for training and testing
    train_list, val_list, test_list = EP_split()
    np.save(exp_dir+'/train_list.npy', train_list)
    np.save(exp_dir+'/val_list.npy', val_list)
    np.save(exp_dir+'/test_list.npy', test_list)
    # print(train_list)

    # define the dataloader
    train_dataset = H5Dataset(train_list, T) # please read the code about H5Dataset when preparing your dataset
    train_dataloader = DataLoader(train_dataset, batch_size=2, # two videos for contrastive learning
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True) # TODO: If you run the code on Windows, please remove num_workers=4.
    
    val_dataset = H5Dataset(val_list, T)  # Assuming validation dataset setup similarly
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # define the model and loss
    # model = PhysNet(S, in_ch=in_ch).to(device).train() #from scratch
    #for finetuning
    model = PhysNet(S, in_ch=in_ch).to(device)
    model.load_state_dict(torch.load('/home/jamesemi/Desktop/james/triage/contrast-phys/demo/model_weights.pt', map_location=device)) #if finetuning, check path and use this.
    model.train()

    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250)

    # define irrelevant power ratio
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # define the optimizer
    opt = optim.AdamW(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    model.train()

    # scheduler = StepLR(opt, step_size=10, gamma=0.1)
    best_val_loss = float('inf')  # Track the best validation loss for saving the best model

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
        print(f'Shape of of train_losses is {len(train_losses)}')

        model.eval()
        with torch.no_grad():
            for imgs in val_dataloader:
                imgs = imgs.to(device)
                model_output = model(imgs)
                val_loss, _, _ = loss_func(model_output)
                val_losses.append(val_loss.item())
                val_loss_total += val_loss.item()

            print(f'Shape of val_losses is {len(val_losses)}')
        torch.save(model.state_dict(), exp_dir + '/epoch%d.pt' % e)
        # torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch{e}_model_{timestamp}.pt'))
        print(f'Epoch {e+1}: Train loss - {loss.item():.4f}; Val loss - {val_loss:.4f}')

        avg_val_loss = val_loss_total / len(val_dataloader)
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, f'best_model_{timestamp}.pt'))

        print(f'Epoch {e+1}: Train loss - {loss.item():.4f}; Val loss - {val_loss:.4f}')
        # torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch{e}_model_{timestamp}.pt'))
        torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch{e}.pt'))

        # LR scheduling - not used rn
        #scheduler.step()

    #smoothing out losses for plotting
    train_cols = len(train_losses) // 10
    val_cols = len(val_losses) // 10
    train_losses_np = np.array(train_losses)
    val_losses_np = np.array(val_losses)
    train_losses_smooth = train_losses_np.reshape(10, train_cols)
    val_losses_smooth = val_losses_np.reshape(10, val_cols)
    # mean_array = reshaped_array.mean(axis=0)
    train_losses_smooth = train_losses_smooth.mean(axis=0)
    val_losses_smooth = val_losses_smooth.mean(axis=0)

    # Plotting losses after training
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_smooth, label='Training Loss') #train_losses / train_losses_smooth
    plt.plot(val_losses_smooth, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(exp_dir, f'loss_plot_{timestamp}.png'))  # Save the plot with timestamp