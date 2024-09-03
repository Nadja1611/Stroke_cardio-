# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:46:52 2024

@author: nadja
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, models
import torch.nn.functional as F
import torchvision.transforms as T
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
from dataset import *
from eval_utils import compute_lesion_f1_score
import argparse
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)


# %%
parser = argparse.ArgumentParser(
    description="Arguments for segmentation network.", add_help=False
)
parser.add_argument(
    "-l",
    "--loss_variant",
    type=str,
    help="which loss variant should be used",
    default="combi",
)
parser.add_argument(
    "-batch_size",
    "--batch_size",
    type=int,
    help="number of prosqueuejection angles sinogram",
    default=16,
)
parser.add_argument(
    "-alpha",
    "--alpha",
    type=float,
    help="weight between BCE and dice",
    default=0.99,
)

parser.add_argument(
    "-m",
    "--method",
    type=str,
    help="which method do we use (baseline, joint,..)",
    default="joint",
)
parser.add_argument(
    "-in",
    "--inputs",
    type=str,
    help="which inputs do we use (DWI, ADC, DWI-ADC,..)",
    default="DWI,ADC,differences",
)

parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="which learning rate should be used",
    default=1e-5,
)
parser.add_argument(
    "-o",
    "--logdir",
    type=str,
    help="directory for log files",
    default="/home/nadja/Stroke/logs",
)
parser.add_argument(
    "-w",
    "--weights_dir",
    type=str,
    help="directory to save model weights",
    default="/home/nadja/Stroke/weights",
)

args = parser.parse_args()

""" explanation loss_variant"""
## combi        - combination of dice loss and balanced crossentropy loss


# Example experiment names
experiment_name = (
    "/experiment_" + args.method + "_" + args.loss_variant + "_" + str(args.alpha)
)
tensorboard_dir = args.logdir + experiment_name
try:
    os.mkdir(tensorboard_dir)
except:
    pass
writer = SummaryWriter(log_dir=tensorboard_dir)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x, y1 = self.up1(x5, x4)
        x, y2 = self.up2(x, x3)
        x, y3 = self.up3(x, x2)
        x, y4 = self.up4(x, x1)
        logits = self.outc(x)
        return F.sigmoid(logits), F.sigmoid(y1), F.sigmoid(y2), F.sigmoid(y3)


class AutoEncoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AutoEncoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x, ds1 = self.up1(x5, x4)
        x, ds2 = self.up2(x, x3)
        x, ds3 = self.up3(x, x2)
        x, ds4 = self.up4(x, x1)
        logits = self.outc(x)
        return logits, ds1, ds2, ds3


class Joint_reconstruction_and_segmentation_dwi_lesion:
    def __init__(
        self,
        learning_rate: float = 5e-5,
        # learning_rate: float = 5e-6,
        device: str = "cuda:0",
        inputs: str = "DWI",
    ):
        self.method = "joint"
        self.mean_FP = []
        self.mean_FN = []
        self.Recall = 0
        self.mean_recall = []
        self.learning_rate = 1e-5
        self.recon_weight = 0.5
        # self.alpha = 0.99
        self.alpha = args.alpha
        self.gamma = 0.5
        self.delta = 0.5
        self.weight = 0.001
        self.lesion_yes_no = []
        # self.inputs = "DWI"
        if args.method == "joint":
            self.dwi_2_adc_net = AutoEncoder(n_channels=1, n_classes=1).to(device)
            self.optimizer_dwi_2_adc = optim.Adam(
                self.dwi_2_adc_net.parameters(), lr=self.learning_rate
            )
            self.adc_2_dwi_net = AutoEncoder(n_channels=1, n_classes=1).to(device)
            self.optimizer_adc_2_dwi = optim.Adam(
                self.adc_2_dwi_net.parameters(), lr=self.learning_rate
            )
        self.segmentation_net = UNet(n_channels=4, n_classes=1).to(device)
        self.optimizer = optim.Adam(
            self.segmentation_net.parameters(), lr=self.learning_rate
        )

        self.device = "cuda:0"
        self.Dice = 0
        self.F1 = 0
        self.F_score = 0
        self.S_score = 0
        self.AP = 0
        self.Dice_isles = 0
        self.inputs = inputs
        self.mean_dice = []
        self.mean_AP = []
        self.mean_dice_isles = []
        self.mean_sens_score = []
        self.mean_fp_score = []
        self.mean_F1 = []
        self.FP = 0
        self.FN = 0
        self.mean_spec = []
        self.Spec = 0
        self.max_mean_spec = 0
        self.max_mean_dice = 0
        self.max_mean_sens = 0
        self.Loss_list = []
        self.seg_loss = []
        self.reco_loss = []
        self.reco_weight = 0.5
        self.batch_size = 32
        # Dataset
        self.train_dataset = StrokeDataset(
            file_path="/home/nadja/Stroke/data2D/train/data.pt"
        )
        self.test_dataset = StrokeDataset(
            file_path="/home/nadja/Stroke/data2D/test/data.pt"
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )

    def init_NW(self, device):
        if self.inputs == "DWI":
            self.segmentation_net = UNet(n_channels=1, n_classes=1).to(device)
        if self.inputs == "DWI + ADC":
            self.segmentation_net = UNet(n_channels=2, n_classes=1).to(device)
        if self.inputs == "DWI + ADC + DWI_ADC":
            self.segmentation_net = UNet(n_channels=3, n_classes=1).to(device)
        if self.inputs == "DWI + ADC + DWI_ADC" and self.method == "joint":
            self.segmentation_net = UNet(n_channels=4, n_classes=1).to(device)
        self.optimizer = optim.Adam(
            self.segmentation_net.parameters(), lr=self.learning_rate
        )

    def define_nonlesion_slices(self, segmentation_mask):
        """function that give back a tensor indicating whether on slice i there is a lesion or not"""
        lesion_yes_no = torch.empty((len(segmentation_mask), 1))
        # if there is no lesion on the slice,
        index = torch.tensor(
            [
                [x]
                for x in range(len(segmentation_mask))
                if torch.max(segmentation_mask[x]) == 0
            ]
        )

        if len(index) > 0:
            lesion_yes_no[index] = 1
        else:
            lesion_yes_no == torch.zeros_like(lesion_yes_no)
        lesion_yes_no = torch.round(lesion_yes_no)

        return lesion_yes_no

    def compute_weight(self):
        """use the labels of the whole dataset and compute imbalance for BCE loss term"""
        shape = self.gt.shape
        self.weight = torch.sum(self.gt) / (shape[2] ** 2 * shape[0])

    def size_prior(self, y_pred):
        """
        Size prior loss.

        Args:
        - y_pred (torch.Tensor): Predicted segmentation mask, shape (batch_size, 1, height, width).

        Returns:
        - size_loss (torch.Tensor): Size prior loss.
        """

        # Convert predicted mask to binary mask
        binary_mask = torch.where(
            y_pred > 0.5, torch.ones_like(y_pred), torch.zeros_like(y_pred)
        )

        # Compute connected components
        connected_components, _ = torch.unique(
            binary_mask.int(), sorted=True, return_inverse=True
        )

        # Count number of pixels in each connected component
        pixel_counts = torch.bincount(connected_components.flatten())

        # Penalize larger connected components
        size_loss = torch.mean(
            pixel_counts[1:]
        )  # Exclude background component (index 0)

        return size_loss

    def symmetry_prior(self, y_true, y_pred):
        """
        Symmetry loss.

        Args:
        - y_true (torch.Tensor): Ground truth segmentation mask, shape (batch_size, 1, height, width).
        - y_pred (torch.Tensor): Predicted segmentation mask, shape (batch_size, 1, height, width).

        Returns:
        - summe (torch.Tensor): Symmetry loss.
        """
        # Threshold the predicted mask
        pred_thresh = y_pred[:, :, :, :, 0]

        # Filter the ground truth
        gt_filtered = 1 - F.max_pool2d(
            y_true[:, :, :, :, 0], kernel_size=5, stride=1, padding=2
        )

        # Filter the predicted mask

        pred_filtered = F.max_pool2d(
            pred_thresh * gt_filtered, kernel_size=5, stride=1, padding=2
        )

        # Flip the filtered predicted mask
        flipped = torch.flip(pred_filtered, dims=[3]) * pred_filtered

        # Calculate mean
        summe = torch.mean(flipped)

        return summe

    def segmentation_step(self, data, data_val):
        gc.collect()

    ########### tversky loss ####################################

    def tversky(self, tp, fn, fp):
        loss2 = 1 - (
            (torch.sum(tp) + 1e-5)
            / (
                torch.sum(tp)
                + self.gamma * torch.sum(fn)
                + self.delta * torch.sum(fp)
                + 1e-5
            )
        )
        return loss2

    def evaluate(self, segmentation_mask, output):
        output = torch.round(output)
        tp = torch.sum(output * segmentation_mask)
        tn = torch.sum((1 - output) * (1 - segmentation_mask))
        fn = torch.sum((1 - output) * (segmentation_mask))
        fp = torch.sum((output) * (1 - segmentation_mask))
        Recall = (tp + 0.0001) / (tp + fn + 0.0001)
        Spec = tn / (tn + fp)
        Dice = 2 * tp / (2 * tp + fn + fp)
        """ Compute f1 score as in isles paper """
        # F1 = compute_lesion_f1_score(
        #    segmentation_mask[:, 0].detach().cpu(), output[:, 0].detach().cpu()
        # )

        im1 = np.asarray(segmentation_mask.detach().cpu()).astype(bool)
        im2 = np.asarray(output.detach().cpu()).astype(bool)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return 1.0

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        dice_isles = 2.0 * intersection.sum() / im_sum

        self.Dice_isles += dice_isles
        self.Dice += Dice.detach().cpu()
        #  self.F1 += F1
        self.FP += fp.detach().cpu()
        self.FN += fn.detach().cpu()
        self.Recall += Recall
        self.Spec += Spec

        del (fn, fp, tn, tp)
        gc.collect()

    ###### loss for joint reconstruction and segmentation ########
    def joint_loss(self, segmentation_mask, output):
        weights = torch.stack(
            [torch.tensor(1 - self.weight), torch.tensor(self.weight)]
        ).to(self.device)
        output = torch.stack([output, 1 - output], axis=-1)
        segmentation_mask = torch.stack(
            [segmentation_mask, 1 - segmentation_mask], axis=-1
        )
        # weights = 1-torch.tensor(self.weight)
        output = torch.clip(output, min=1e-6)
        loss1 = -torch.sum(segmentation_mask * torch.log(output) * weights, axis=-1)
        loss1 = torch.mean(loss1)
        """tversky preperation"""
        y_true_f = torch.flatten(segmentation_mask[:, :, :, :, :1])
        y_pred_f = torch.flatten(output[:, :, :, :, :1])
        fp = (1 - y_true_f) * y_pred_f
        fn = (1 - y_pred_f) * y_true_f
        tp = y_pred_f * y_true_f
        r = torch.randint(0, 100, (1,))
        if r == 0:
            print("bce: " + str(loss1))
            print("dice: " + str(self.tversky(tp, fn, fp)))
            print("sym: " + str(self.symmetry_prior(segmentation_mask, output)))
        loss = (self.alpha * loss1) + (1 - self.alpha) * (self.tversky(tp, fn, fp))
        del (tp, fp, fn, y_true_f, y_pred_f)
        gc.collect()
        return loss

    """'as input, we have the real ADC, the reconstructed ADC,and  the segmentation mask """

    def DWI2ADC_loss_only_difference(
        self, adc_input, output_dwi_2_adc, is_lesion_on_slice
    ):
        shape = adc_input.shape
        loss = torch.sum(
            torch.round(is_lesion_on_slice).unsqueeze(-1).unsqueeze(-1)
            * torch.abs(adc_input - output_dwi_2_adc)
        ) / (torch.sum((shape[2] ** 2 * torch.sum(is_lesion_on_slice))) + 1e-5)
        return loss

    """'as input, we have the real DWI, the reconstructed DWI,and  the segmentation mask """

    def ADC2DWI_loss_only_difference(
        self, dwi_input, output_adc_2_dwi, is_lesion_on_slice
    ):
        shape = dwi_input.shape
        loss = torch.sum(
            torch.round(is_lesion_on_slice).unsqueeze(-1).unsqueeze(-1)
            * torch.abs(dwi_input - output_adc_2_dwi)
        ) / (torch.sum((shape[2] ** 2 * torch.sum((is_lesion_on_slice)))) + 1e-5)
        return loss


mynet = Joint_reconstruction_and_segmentation_dwi_lesion(inputs="DWI + ADC + DWI_ADC")
mynet.method = "joint"
mynet.init_NW(device=mynet.device)


N_epochs = 100


for epoch in range(N_epochs):
    running_loss = 0
    running_L2_loss = 0

    losses = []
    i = 0
    running_loss = 0.0

    # (N2NR.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
    for batch, data in enumerate(mynet.train_dataloader):
        mynet.segmentation_net.train()
        if args.method == "joint":
            mynet.dwi_2_adc_net.train()
            mynet.adc_2_dwi_net.train()

        """ specify the inputs of the networks """
        dwi_input, adc_input, segmentation_mask = (
            data[:, 0:1, :, :, 0].to(mynet.device),
            data[:, 1:2, :, :, 0].to(mynet.device),
            data[:, 2:3, :, :, 0].to(mynet.device),
        )

        # print(input_x.shape)
        mynet.optimizer.zero_grad()
        # initialize optim for dwi to adc nw
        if args.method == "joint":
            mynet.optimizer_dwi_2_adc.zero_grad()
            mynet.optimizer_adc_2_dwi.zero_grad()

        # gt for segmentation
        # gt for ADC to dwi nw training
        is_lesion_on_slice = mynet.define_nonlesion_slices(segmentation_mask).to(
            mynet.device
        )
        # segmentation predictions
        output_dwi_2_adc, ds1_adc, ds2_adc, ds3_adc = mynet.dwi_2_adc_net(
            dwi_input.float().to(mynet.device)
        )
        output_adc_2_dwi, ds1_dwi, ds2_dwi, ds3_dwi = mynet.adc_2_dwi_net(
            adc_input.float().to(mynet.device)
        )

        output, output1, output2, output3 = mynet.segmentation_net(
            torch.cat(
                [
                    dwi_input,
                    adc_input,
                    torch.abs(output_dwi_2_adc - adc_input.float().to(mynet.device)),
                    torch.abs(output_adc_2_dwi - dwi_input.float().to(mynet.device)),
                ],
                axis=1,
            )
        )
        loss_seg = (
            mynet.joint_loss(segmentation_mask, output)
            + 0.125 * mynet.joint_loss(F.max_pool2d(segmentation_mask, (8, 8)), output1)
            + 0.25 * mynet.joint_loss(F.max_pool2d(segmentation_mask, (4, 4)), output2)
            + 0.5 * mynet.joint_loss(F.max_pool2d(segmentation_mask, (2, 2)), output3)
        )

        #### if there are lesions on at least one slice################
        if args.method == "joint":
            if torch.max(torch.round(is_lesion_on_slice)) > 0:
                loss_reco = mynet.DWI2ADC_loss_only_difference(
                    adc_input, output_dwi_2_adc, is_lesion_on_slice
                )
                loss_reco2 = mynet.ADC2DWI_loss_only_difference(
                    dwi_input, output_adc_2_dwi, is_lesion_on_slice
                )
                #############----deep supervision----##################
                loss_reco_ds1 = mynet.DWI2ADC_loss_only_difference(
                    F.avg_pool2d(adc_input, (8, 8)), ds1_adc, is_lesion_on_slice
                )
                loss_reco2_ds1 = mynet.ADC2DWI_loss_only_difference(
                    F.avg_pool2d(dwi_input, (8, 8)), ds1_dwi, is_lesion_on_slice
                )

                loss_reco_ds2 = mynet.DWI2ADC_loss_only_difference(
                    F.avg_pool2d(adc_input, (4, 4)), ds2_adc, is_lesion_on_slice
                )
                loss_reco2_ds2 = mynet.ADC2DWI_loss_only_difference(
                    F.avg_pool2d(dwi_input, (4, 4)), ds2_dwi, is_lesion_on_slice
                )

                loss_reco_ds3 = mynet.DWI2ADC_loss_only_difference(
                    F.avg_pool2d(adc_input, (2, 2)), ds3_adc, is_lesion_on_slice
                )
                loss_reco2_ds3 = mynet.ADC2DWI_loss_only_difference(
                    F.avg_pool2d(dwi_input, (2, 2)), ds3_dwi, is_lesion_on_slice
                )

                loss_total = loss_seg + 0.5 * (
                    loss_reco2
                    + loss_reco
                    + 0.125 * loss_reco_ds1
                    + 0.125 * loss_reco2_ds1
                    + 0.25 * loss_reco_ds2
                    + 0.25 * loss_reco2_ds2
                    + 0.5 * loss_reco_ds3
                    + 0.5 * loss_reco2_ds3
                )
                loss_total.backward()
                mynet.optimizer.step()
                mynet.optimizer_dwi_2_adc.step()
                mynet.optimizer_adc_2_dwi.step()
            else:
                loss_total = loss_seg
                loss_total.backward()
                mynet.optimizer.step()

        else:
            loss_total = loss_seg
            loss_total.backward()
            mynet.optimizer.step()

    with torch.no_grad():
        running_loss += loss_total.item()
        if i % 10 == 9:
            print(
                "[Epoque : %d, iteration: %5d] loss: %.3f"
                % (epoch + 1, i + 1, running_loss / 10),
                flush=True,
            )
            mynet.Loss_list.append(running_loss / 10)

            running_loss = 0.0

        i += 1

    with torch.no_grad():
        mynet.segmentation_net.eval()
        if args.method == "joint":
            mynet.dwi_2_adc_net.eval()
            mynet.adc_2_dwi_net.eval()
        for features in mynet.test_dataloader:
            if mynet.inputs == "DWI + ADC + DWI_ADC":
                dwi_input, adc_input, segmentation_mask = (
                    data[:, 0:1, :, :, 0].to(mynet.device),
                    data[:, 1:2, :, :, 0].to(mynet.device),
                    data[:, 2:3, :, :, 0].to(mynet.device),
                )
            if args.method == "joint":
                output_dwi_2_adc, ds1_adc, ds2_adc, ds3_adc = mynet.dwi_2_adc_net(
                    dwi_input.float().to(mynet.device)
                )
                output_adc_2_dwi, ds1_dwi, ds2_dwi, ds3_dwi = mynet.adc_2_dwi_net(
                    adc_input.float().to(mynet.device)
                )
            output, output1, output2, output3 = mynet.segmentation_net(
                torch.cat(
                    [
                        dwi_input,
                        adc_input,
                        torch.abs(
                            output_dwi_2_adc - adc_input.float().to(mynet.device)
                        ),
                        torch.abs(
                            output_adc_2_dwi - dwi_input.float().to(mynet.device)
                        ),
                    ],
                    axis=1,
                )
            )
            mynet.evaluate(segmentation_mask.to(mynet.device), output)

    mynet.mean_dice.append(
        ((torch.tensor(mynet.Dice)) / len(mynet.test_dataloader)).detach().cpu()
    )
    mynet.mean_dice_isles.append(
        ((torch.tensor(mynet.Dice_isles)) / len(mynet.test_dataloader)).detach().cpu()
    )
    # mynet.mean_F1.append(
    #   ((torch.tensor(mynet.F1)) / len(mynet.test_dataloader)).detach().cpu()
    # )
    mynet.mean_recall.append(
        ((torch.tensor(mynet.Recall)) / len(mynet.test_dataloader)).detach().cpu()
    )
    mynet.mean_spec.append(
        ((torch.tensor(mynet.Spec)) / len(mynet.test_dataloader)).detach().cpu()
    )
    mynet.mean_FP.append(
        ((torch.tensor(mynet.FP)) // len(mynet.test_dataloader)).detach().cpu()
    )
    mynet.mean_FN.append(
        ((torch.tensor(mynet.FN)) / len(mynet.test_dataloader)).detach().cpu()
    )

    writer.add_scalar("Loss/epoch", running_loss / len(mynet.train_dataloader), epoch)
    writer.add_scalar(
        "Dice",
        (torch.tensor(mynet.Dice) / len(mynet.test_dataloader)).detach().cpu(),
        epoch,
    )

    writer.close()
    plt.subplot(2, 2, 1)
    plt.imshow(dwi_input[2][0].detach().cpu(), cmap="gray")
    plt.subplot(2, 2, 2)
    plt.imshow(segmentation_mask[2][0].detach().cpu(), cmap="gray")
    plt.subplot(2, 2, 3)
    plt.imshow(output[2][0].detach().cpu(), cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(
        torch.abs(output_adc_2_dwi - dwi_input.float()).detach().cpu()[2][0],
        cmap="gray",
    )
    plt.savefig("/home/nadja/Stroke/images/output" + str(epoch) + ".png")

    if mynet.mean_dice_isles[-1] > mynet.max_mean_dice:
        mynet.max_mean_dice = mynet.mean_dice_isles[-1]
        name_weights = (
            "best_dice_weights_joint_ds_normalized_"
            + str(mynet.reco_weight)
            + "_"
            + mynet.inputs
            + ".hdf5"
        )
        name_weights_dwi2adc = (
            "best_dice_weights_joint_ds_normalized_"
            + str(mynet.reco_weight)
            + "_dwi2adc_"
            + mynet.inputs
            + ".hdf5"
        )
        name_weights_adc2dwi = (
            "best_dice_weights_joint_ds_normalized_"
            + str(mynet.reco_weight)
            + "_adc2dwi_"
            + mynet.inputs
            + ".hdf5"
        )

        torch.save(
            mynet.segmentation_net.state_dict(),
            args.weights_dir + "/" + name_weights,
        )
        torch.save(
            mynet.dwi_2_adc_net.state_dict(),
            args.weights_dir + "/" + name_weights_dwi2adc,
        )

        torch.save(
            mynet.adc_2_dwi_net.state_dict(),
            args.weights_dir + "/" + name_weights_adc2dwi,
        )

        print("saved weights")
    if epoch > 50 and mynet.mean_recall[-1] > mynet.max_mean_sens:
        mynet.max_mean_sens = mynet.mean_recall[-1]
        name_weights = (
            "best_recall_weights_joint_ds_normalized_"
            + str(mynet.reco_weight)
            + "_"
            + mynet.inputs
            + ".hdf5"
        )
        name_weights_dwi2adc = (
            "best_recall_weights_joint_ds_normalized_"
            + str(mynet.reco_weight)
            + "_dwi2adc_"
            + mynet.inputs
            + ".hdf5"
        )
        name_weights_adc2dwi = (
            "best_recall_weights_joint_ds_normalized_"
            + str(mynet.reco_weight)
            + "_adc2dwi_"
            + mynet.inputs
            + ".hdf5"
        )

        torch.save(
            mynet.segmentation_net.state_dict(),
            args.weights_dir + "/" + name_weights,
        )

        torch.save(
            mynet.dwi_2_adc_net.state_dict(),
            args.weights_dir + "/" + name_weights_dwi2adc,
        )

        torch.save(
            mynet.adc_2_dwi_net.state_dict(),
            args.weights_dir + "/" + name_weights_adc2dwi,
        )

        print("saved weights")

    if mynet.mean_spec[-1] > mynet.max_mean_spec:
        mynet.max_mean_spec = mynet.mean_spec[-1]
        name_weights = (
            "best_spec_weights_joint_ds_normalized_"
            + str(mynet.reco_weight)
            + "_"
            + mynet.inputs
            + ".hdf5"
        )
        name_weights_dwi2adc = (
            "best_spec_weights_joint_ds_normalized_"
            + str(mynet.reco_weight)
            + "_dwi2adc_"
            + mynet.inputs
            + ".hdf5"
        )
        name_weights_adc2dwi = (
            "best_spec_weights_joint_ds_normalized_"
            + str(mynet.reco_weight)
            + "_adc2dwi_"
            + mynet.inputs
            + ".hdf5"
        )

        torch.save(
            mynet.segmentation_net.state_dict(),
            args.weights_dir + "/" + name_weights,
        )
        torch.save(
            mynet.dwi_2_adc_net.state_dict(),
            args.weights_dir + "/" + name_weights_dwi2adc,
        )

        torch.save(
            mynet.adc_2_dwi_net.state_dict(),
            args.weights_dir + "/" + name_weights_adc2dwi,
        )

        print("saved weights")

    mynet.Dice = 0
    mynet.F1 = 0
    mynet.Dice_isles = 0
    mynet.Recall = 0
    mynet.Spec = 0
    mynet.FP = 0
    mynet.FN = 0
    mynet.AP = 0
    mynet.FP_score = 0
    mynet.S_score = 0

    del output
    del segmentation_mask
    gc.collect()

    np.savez_compressed(
        "curves_joint_DS_reco_0.1_normalized.npz",
        dice=mynet.mean_dice_isles,
        fp=mynet.mean_spec,
        recall=mynet.mean_recall,
    )
