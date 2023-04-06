import os
import pandas as pd
import numpy as np
from PIL import Image

def get_dataset_stats(csv_filename):
    files = pd.read_csv(csv_filename, header=None, skiprows=[0])
    dataset_array_list = []
    for index, file in files.iterrows():
        image_name = file[4]
        image = np.array(Image.open(image_name, "r"))
        dataset_array_list.append(image)

    dataset_numpy_array = np.stack(dataset_array_list, axis=0)
    mean_over_entire_dataset = dataset_numpy_array.mean(axis=(0,1,2))
    std_over_entire_dataset = dataset_numpy_array.std(axis=(0,1,2))
    return mean_over_entire_dataset, std_over_entire_dataset;

