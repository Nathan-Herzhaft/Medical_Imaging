# %%
# 1. Import useful libraries

import logging
import os
import sys
import tempfile
from glob import glob

import torch
from PIL import Image
from torch.utils.data import DataLoader

import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
)
from monai.visualize import plot_2d_or_3d_image

print('Modules imported successfully')










# %%
# 2. Load Data

def generate_input(data_folder):
    """generate images and segmented images from the folder

    Args:
        data_folder (str): path of the data folder from where to extract the data

    Returns:
        images (list): sorted paths of the input images
        segs (list): sorted paths of the segmented images
    """    
    
    images = sorted(glob(os.path.join(data_folder, "*_scan.tif")))
    segs = sorted(glob(os.path.join(data_folder, "*_mask.tif")))
    return images, segs

images, segs = generate_input('data/train')
print(f'Number of files : {len(images)}')










# %%
# 3. Define loader classes

def get_transforms(mode) :
    """Define transformations to apply to data. Specific transformation are applied to training inputs

    Args:
        mode (str): mode of data : 'train' or else

    Returns:
        train_transforms (Compose): Composition of transformations
    """
        
    trans = [
        LoadImage(image_only=True), 
        AddChannel(), 
        ScaleIntensity()
    ]
    if mode =='train' : 
        trans.extend([
            RandSpatialCrop((96, 96), random_size=False),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ])
    trans.extend([EnsureType()])
    return Compose(trans)

train_transforms, val_transforms = get_transforms('train'), get_transforms('val')


def get_loaders(images, segs, train_transforms, val_transforms, n_train, n_val) :
    """Define dataset and dataloader classes for the model

    Args:
        images (list): sorted paths of the input images
        segs (list): sorted paths of the segmented images
        train_transforms (Compose): Composition of transformations for training
        val_transforms (Compose): Composition of transformations for validation
        n_train (int): number of inputs loaded for training
        n_val (int): number of inputs loaded for validation

    Returns:
        train_loader (DataLaoder): Loader for training
        val_loader (DataLaoder): Loader for validation
        train_ds (DataSet): Dataset for training
        val_ds (DataSet): Dataset for validation
    """
            
    train_ds = ArrayDataset(images[:n_train], train_transforms, segs[:n_train], train_transforms)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    
    val_ds = ArrayDataset(images[-n_val:], val_transforms, segs[-n_val:], val_transforms)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    return train_loader, val_loader, train_ds, val_ds

train_loader, val_loader, train_ds, val_ds = get_loaders(images, segs, train_transforms, val_transforms, 4000, 1000)








# %%
# 4. Train the model

def training() :
    """Launch the training, supposing DataLoaders and Datasets have already been defined using previous functions"""
        
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create BasicUNet, DiceLoss and Adam optimizer
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()

    for epoch in range(10):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{10}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0], batch_data[1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"batch {step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0], val_data[1]
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model, "2D_Nerve_Segmentation.pt")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

training()
# %%
