"""This file contains multiple helper functions to plot the data. """
from matplotlib import pyplot as plt
import torch

import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

def display_tensor_data(tensor, remap=None):
    """this function displays a tensor as an image

    Args:
        tensor (tensor): tensor to be displayed
        remap (callable, optional): function to remap the tensor. Defaults to None.
    """    
    

    if remap != None:
        tensor = remap(tensor)
    fig, (ax) = plt.subplots(1)
    pcm = ax.imshow(tensor.permute(1,2,0))
    fig.colorbar(pcm, ax=ax) #ticks=[-1, 0, 1])
    plt.show()

def compute_error(ground_truth_depth, depth_output):
    """Computes the error between the ground truth and predictions under threshold of 0.6"""
    error = torch.abs(ground_truth_depth - depth_output);
    thr = 0.6;
    error[error < thr] = 0
    return error

def display_error_plots(ground_truth_depth, depth_output):
    """Displays the ground truth, output depth and the error plot

    Args:
        ground_truth_depth (tensor): tensor containing the ground truth depth
        depth_output (tensor): tensor containing the output depth
    """    

    error = compute_error(ground_truth_depth, depth_output)

    min_scale = min(ground_truth_depth.min(), depth_output.min())
    max_scale = max(ground_truth_depth.max(), depth_output.max())
    if ground_truth_depth.isnan().any():
        temp = ground_truth_depth
        temp[temp.isnan()] = 3; #TODO figure out how to set
        min_scale = min(temp.min(), depth_output.min())
        max_scale = max(temp.max(), depth_output.max())

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 16))
    pcm = axs[0].imshow(depth_output.permute(1, 2, 0), vmax=max_scale, vmin=min_scale)
    axs[0].set_xlabel("Model Output dDpthmap in meters")
    fig.colorbar(pcm, ax=axs[0], fraction=0.046, pad=0.04)
    pcm2 = axs[1].imshow(ground_truth_depth.permute(1, 2, 0), vmax=max_scale, vmin=min_scale)
    axs[1].set_xlabel("Ground Truth Depthmap in meters")
    fig.colorbar(pcm2, ax=axs[1], fraction=0.046, pad=0.04)
    pcm3 = axs[2].imshow(error.permute(1, 2, 0))
    axs[2].set_xlabel("L1 error thresholded at 0.6 m")
    fig.colorbar(pcm3, ax=axs[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()



def plot(ground_truth_depth, depth_output, rgb_image, plot_name=None):
    """Plots the ground truth, depth output, rgb image and segmentation

    Args:
        ground_truth_depth (tensor): tensor containing the ground truth depth
        depth_output (tensor): tensor containing the output depth
        rgb_image (tensor): tensor containing the rgb image
        plot_name (str, optional): name of the plot. Defaults to None.
    """    

    error = compute_error(ground_truth_depth, depth_output)

    min_scale = min(ground_truth_depth.min(), depth_output.min())
    max_scale = max(ground_truth_depth.max(), depth_output.max())
    if ground_truth_depth.isnan().any():
        temp = ground_truth_depth
        temp[temp.isnan()] = 3; #TODO: set correct value
        min_scale = min(temp.min(), depth_output.min())
        max_scale = max(temp.max(), depth_output.max())

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 16))
    axs[0].imshow(rgb_image.permute(1, 2, 0))
    axs[0].set_xlabel("Input Image")
    pcm = axs[1].imshow(ground_truth_depth.permute(1, 2, 0), vmax=max_scale, vmin=min_scale)
    fig.colorbar(pcm, ax=axs[1], fraction=0.046, pad=0.04)
    axs[1].set_xlabel("Ground Truth Depthmap in meters")
    pcm2 = axs[2].imshow(depth_output.permute(1, 2, 0), vmax=max_scale, vmin=min_scale)
    fig.colorbar(pcm2, ax=axs[2], fraction=0.046, pad=0.04)
    axs[2].set_xlabel("Model Output Depthmap in meters")
    pcm3 = axs[3].imshow(error.permute(1, 2, 0))
    fig.colorbar(pcm3, ax=axs[3], fraction=0.046, pad=0.04)
    axs[3].set_xlabel("L1 error thresholded at 0.6 m")
    fig.tight_layout()
    plt.show()
    if plot_name is not None:
        fig.savefig(os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH']+"../plots/validation_error_plots/"+plot_name+".png", bbox_inches='tight')

def display_tensor_data_many(tensors, remap=None, titles=None, figsize=(15, 15), fontsize=12, plot_name=None):
    """Displays many tensor data

    Args:
        tensors (list): list of tensors to display
        remap (list, optional): list of functions to remap the tensors. Defaults to None.   
        titles (list, optional): list of titles for the plots. Defaults to None.
        figsize (tuple, optional): tuple containing the figure size. Defaults to (15, 15).
        fontsize (int, optional): fontsize for the titles. Defaults to 12.
        plot_name (str, optional): name of the plot. Defaults to None.
    """    

    if titles is None:
        titles = ["Model Prediction", "Ground Truth", "RGB Input"]
    nb = len(titles)
    fig, axes = plt.subplots(1, nb, figsize=figsize)
    if remap is not None:
        for i in range(nb):
            if remap[i] is not None:
                tensors[i] = remap[i](tensors[i])

    for i, ax in enumerate(axes):
        if(titles[i] == "Segmentation Mask as Grayscale Image"):
            pcm = ax.imshow(tensors[i].permute(1,2,0), cmap="gray")
        else:
            pcm = ax.imshow(tensors[i].permute(1, 2, 0))
        ax.set_title(titles[i], fontsize=fontsize)
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.1)
        #fig.colorbar(pcm, cax=cax)  # ticks=[-1, 0, 1])

    fig.tight_layout()
    plt.show()

    if plot_name is not None:
        fig.savefig(os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH']+"../plots/validation_error_plots/"+plot_name+".png", bbox_inches='tight')

def display_image_pairs(rgb_image_tensor, segmentation):
    """Displays the rgb image and the segmentation mask

    Args:
        rgb_image_tensor (tensor): tensor containing the rgb image
        segmentation (tensor): tensor containing the segmentation mask
    """    

    fig, axs = plt.subplots(2)
    axs[0].imshow(rgb_image_tensor.permute(1, 2, 0))
    axs[1].imshow(segmentation)
    plt.show()

def denormalize(image_input):
    """Function used to undo the normalization done during the data loading.

    Args:
        image_input (tensor): tensor containing the image to denormalize

    Returns:
        tensor : denormalized image
    """    
    mean, std = [0.53277088, 0.49348648, 0.45927282], [0.238986, 0.23546355, 0.24486044]
    image = image_input.clone().detach()
    for channel in range(0, image.shape[0]):
        image[channel, :, :].mul_(std[channel]).add_(mean[channel])
    return image


def plot_training_progress(dataframe, title=None, plot_name=None):
    """Plot used during internal training to see the different loss evolutions.

    Args:
        dataframe (pandas datafram): dataframe containing the loss values
        title (str, optional): title of the plot. Defaults to None.
        plot_name (str, optional): name of the plot. Defaults to None.
    """    
    fig = plt.figure()
    ax = fig.gca()
    x = list(range(dataframe.shape[0]))
    x_ticks = list(range(0, dataframe.shape[0], 2))
    ax.set_xticks(x_ticks)
    line, = ax.plot(x, dataframe["loss_depth"], label="loss_depth")
    # line.set_label("loss_depth")
    line, = ax.plot(x, dataframe["loss_dx"], label="loss_dx")
    # line.set_label("loss_dx")
    line, = ax.plot(x, dataframe["loss_dy"], label="loss_dy")
    # line.set_label("loss_dy")
    line, = ax.plot(x, dataframe["loss_normal"], label="loss_normal")
    # line.set_label("loss_normal")
    line, = ax.plot(x, dataframe["loss"], label="loss")
    ax.set_xlabel("Epoch")
    if title is not None:
        ax.set_title(title)

    # line.set_label("loss_normal")
    ax.legend()
    plt.show()

    if plot_name is not None:
        fig.savefig(os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH']+"../plots/training_progress_plots/"+plot_name+".png", bbox_inches='tight')

