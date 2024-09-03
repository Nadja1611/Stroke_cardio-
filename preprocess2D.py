import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn import init, ReflectionPad2d, ZeroPad2d
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from unet_blocks_ds import *
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import gc
from scipy.ndimage.measurements import label
from functions import *
import os
from model_utils import define_inputs
import nibabel as nib

torch.manual_seed(0)

X = []
Y = []
L = []
patients = os.listdir("/scratch/nadja/ISLES-2022/")

for pat in sorted(patients):
    if pat.startswith("sub"):
        inputdir_train = "/scratch/nadja/ISLES-2022/" + pat
        for sub in os.listdir(inputdir_train + "/ses-0001/dwi"):
            if sub.endswith("dwi.nii.gz"):
                x = np.expand_dims(
                    (nib.load(inputdir_train + "/ses-0001/dwi/" + sub).get_fdata()),
                    axis=-1,
                ).transpose(2, 0, 1, 3)
                if x.shape[1] != 112:
                    x = resize(x, (x.shape[0], 112, 112, x.shape[-1]))
                X.append(x)
            if sub.endswith("adc.nii.gz"):
                y = np.expand_dims(
                    (nib.load(inputdir_train + "/ses-0001/dwi/" + sub).get_fdata()),
                    axis=-1,
                ).transpose(2, 0, 1, 3)
                if y.shape[1] != 112:
                    y = resize(y, (y.shape[0], 112, 112, y.shape[-1]))
                Y.append(y)

patients = os.listdir("/scratch/nadja/ISLES-2022/derivatives/")
for pat in sorted(patients):
    if pat.startswith("sub"):
        inputdir_train = "/scratch/nadja/ISLES-2022/derivatives/" + pat + "/ses-0001/"
        for sub in os.listdir(inputdir_train):
            if sub.endswith(".nii.gz"):
                x = np.expand_dims(
                    (nib.load(inputdir_train + sub).get_fdata()), axis=-1
                ).transpose(2, 0, 1, 3)
                if x.shape[1] != 112:
                    x = resize(x, (x.shape[0], 112, 112, x.shape[-1]))
                L.append(x)


X = torch.tensor(np.concatenate(X, 0)).unsqueeze(1)
Y = torch.tensor(np.concatenate(Y, 0)).unsqueeze(1)
L = torch.tensor(np.concatenate(L, 0)[:, :, :, :1]).unsqueeze(1)
data = torch.cat([X, Y, L], axis=1).float()
print(len(data))
print(data.shape)
""" Now, we save slice by slice for later training 2D, in a way, that channel 0: DWI, 1:ADC, 2:GT """
data_dict = {f"pat_{i}": data[i] for i in range(int(np.round(0.8 * len(data))))}
torch.save(data_dict, "/home/nadja/Stroke/data2D/train/data.pt")
data_dict = {
    f"pat_{i}": data[i] for i in range(int(np.round(0.8 * len(data))), len(data))
}
torch.save(data_dict, "/home/nadja/Stroke/data2D/test/data.pt")
