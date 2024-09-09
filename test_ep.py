import numpy as np
import h5py
import torch
from PhysNetModel import PhysNet
from utils_data import *
from utils_sig import *
from sacred import Experiment
from sacred.observers import FileStorageObserver
import json
from tqdm import tqdm  # Import tqdm
from utils_sig import hr_fft, normalize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error


ex = Experiment('model_pred', save_git_info=False)

@ex.config
def my_config():
    e = 29 # the model checkpoint at epoch e
    train_exp_num = 5 # the training experiment number
    train_exp_dir = './results/%d'%train_exp_num # training experiment directory
    time_interval = 30 # get rppg for 30s video clips, too long clips might cause out of memory
    ex.observers.append(FileStorageObserver(train_exp_dir))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

sns.set(style="whitegrid")

def plot_band_altman(pred_hr, gt_hr, label="Band-Altman Plot"):
    means = np.mean([pred_hr, gt_hr], axis=0)
    diffs = pred_hr - gt_hr
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs)

    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x=means, y=diffs, color='blue', edgecolor='w', s=100)
    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean diff')
    plt.axhline(mean_diff + 1.96 * sd_diff, color='green', linestyle='--', label='1.96 SD')
    plt.axhline(mean_diff - 1.96 * sd_diff, color='green', linestyle='--')
    plt.axhline(0, color='black', linestyle='-', label='y=x (zero diff)')
    plt.xlabel('Mean HR [(predicted + ground truth) / 2]')
    plt.ylabel('Diff HR [(predicted - ground truth)]')
    plt.title(label)
    plt.legend()
    plt.show()

sns.set(style="whitegrid")

def plot_hr_comparison(pred_hr, gt_hr, label="HR Comparison Plot"):
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x=gt_hr, y=pred_hr, color='blue', edgecolor='w', s=100)
    plt.plot([min(gt_hr), max(gt_hr)], [min(gt_hr), max(gt_hr)], 'r--', label="y=x line")
    plt.xlabel('Ground Truth HR')
    plt.ylabel('Predicted HR')
    plt.title(label)
    plt.legend()
    plt.show()

@ex.automain
def my_main(_run, e, train_exp_dir, device, time_interval, train_exp_num):

    train_list, val_list, test_list = EP_split()
    # load test file paths
    # test_list = list(np.load(train_exp_dir + '/test_list.npy'))
    # pred_exp_dir = train_exp_dir + '/%d'%(int(_run._id)) # prediction experiment directory
    pred_exp_dir = '/home/jamesemi/Desktop/james/triage/contrast-phys/results/testing' #using a new folder for easier access.

    pred_hrs = []
    gt_hrs = []
    subject_sesh_list = []
    fs = 30  # Define the sampling frequency

    with open(train_exp_dir+'/config.json') as f:
        config_train = json.load(f)

    # model = PhysNet(config_train['S'], config_train['in_ch']).to(device).eval()
    model = PhysNet().to(device).eval()
    model.load_state_dict(torch.load(train_exp_dir+'/epoch%d.pt'%(e), map_location=device)) # load weights to the model
    ## for one check, load standard cphys model (from github), and test. Done?
    # std_path = '/home/jamesemi/Desktop/james/triage/contrast-phys/demo/model_weights.pt'
    # model.load_state_dict(torch.load(std_path, map_location=device)) # load weights to the model
    ## load ckpt trained on EP data and test. Done?

    @torch.no_grad()
    def dl_model(imgs_clip):
        # model inference
        img_batch = imgs_clip
        img_batch = img_batch.transpose((3,0,1,2))
        img_batch = img_batch[np.newaxis].astype('float32')
        img_batch = torch.tensor(img_batch).to(device)

        rppg = model(img_batch)[:,-1, :]
        rppg = rppg[0].detach().cpu().numpy()
        return rppg

    for h5_path in tqdm(test_list, desc="Processing .h5 files"):
        # print(f'Test list is: {test_list}') #good for debugging

        with h5py.File(h5_path, 'r') as f:
            if 'bvp' in f:
                print(list(f.keys()))

                imgs = f['imgs']
                bvp = f['bvp']
                # bvppeak = f['bvp_peak']
                fs = config_train['fs']

                duration = np.min([imgs.shape[0], bvp.shape[0]]) / fs
                num_blocks = int(duration // time_interval)

                rppg_list = []
                bvp_list = []
                # bvppeak_list = []

                for b in range(num_blocks):
                    rppg_clip = dl_model(imgs[b*time_interval*fs:(b+1)*time_interval*fs])
                    rppg_list.append(rppg_clip)

                    bvp_list.append(bvp[b*time_interval*fs:(b+1)*time_interval*fs])
                    # bvppeak_list.append(bvppeak[b*time_interval*fs:(b+1)*time_interval*fs])

                rppg_list = np.array(rppg_list)
                bvp_list = np.array(bvp_list)
                # bvppeak_list = np.array(bvppeak_list)
                # results = {'rppg_list': rppg_list, 'bvp_list': bvp_list, 'bvppeak_list':bvppeak_list}
                results = {'rppg_list': rppg_list, 'bvp_list': bvp_list}
                np.save(pred_exp_dir+'/'+h5_path.split('/')[-1][:-3], results)
                results_rppg = results['rppg_list']
                results_bvp = results['bvp_list']
                print(f'Shape of rppg_list in results is {results_rppg.shape} and that of bvp_list is {results_bvp.shape}')

                # bvp = f['bvp'][:]
                bvp = results['bvp_list']
                # rppg = f['rppg_list'][:]
                rppg = results['rppg_list']
                # probably need to replace the above with data pulled from results

                # Assuming rppg is a single array of rPPG values; adjust if it's not.
                # bvp = normalize(bvp)
                # rppg = normalize(rppg)
                
                final_bvp = butter_bandpass(bvp, lowcut=0.6, highcut=4, fs=30)
                final_rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=30)
                gt_hr, _, _ = hr_fft(final_bvp, fs)
                pred_hr, _, _ = hr_fft(final_rppg, fs)

                pred_hrs.append(pred_hr)
                print(f'Predicted HR here for {f} is {pred_hr} bpm')
                gt_hrs.append(gt_hr)
                print(f'GT HR here for {f} is {gt_hr} bpm')
                subject_sesh_list.append(h5_path)

    # Calculate Mean Absolute Error (MAE)
    mae_value = mean_absolute_error(gt_hrs, pred_hrs)
    print(f'Mean Absolute Error (MAE) between predicted and ground truth HR: {mae_value}')

    # Save predictions, ground truth, and subject_sesh to CSV
    csv_filename = pred_exp_dir + f'/ep_test_results_expnum{train_exp_num}_epoch{e}.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Subject_Session', 'Predicted_HR', 'Ground_Truth_HR', 'Model_Checkpoint'])
        for sesh, pred, gt in zip(subject_sesh_list, pred_hrs, gt_hrs):
            writer.writerow([sesh, pred, gt, f'/results/expnum{train_exp_num}_epoch{e}'])
    # TODO: Add ability to get MAE value
    # TODO: Save the pred_hr, gt_hr alongside the subject_sesh value in csv file.
    # plot_band_altman(np.array(pred_hrs), np.array(gt_hrs))
    plot_hr_comparison(np.array(pred_hrs), np.array(gt_hrs))
