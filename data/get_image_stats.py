import os
import pandas as pd
import numpy as np
import h5py
from ../model/Revisiting_Single_Depth_Estimation-master/models/data_loader_test import remap_data

def get_image_mean(filename):
    remapped_data = remap data(filename)
    return remapped_data.mean(axis=(0,1)) #shape (width, height, channels)


def get_dataset_stats(csv_filename):
    files = pd.read_csv(csv_filename, header=None)
    dataset_array_list = [];
    for file in range(0:files.shape[0]):
        image_name = files.ix[file, 0]
        image = h5py.File(image_name, "r")["dataset"][()]
        dataset_array_list.append(remap_data(image))

    dataset_numpy_array = np.stack(dataset_array_list, axis=0)
    mean_over_entire_dataset = dataset_numpy_array.mean(axis=(0,1,2))
    std_over_entire_dataset = dataset_numpy_array.std(axos=(0,1,2))

