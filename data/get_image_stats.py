import os
import pandas as pd
import numpy as np
from PIL import Image
import math

def get_dataset_stats(csv_filename):
    files = pd.read_csv(csv_filename)
    dataset_array_list = []
    for file in files["ToneMapped"]:
        image_name = file
        image = np.array(Image.open(image_name, "r"))
        dataset_array_list.append(image)

    dataset_numpy_array = np.stack(dataset_array_list, axis=0)
    mean_over_entire_dataset = dataset_numpy_array.mean(axis=(0,1,2))
    std_over_entire_dataset = dataset_numpy_array.std(axis=(0,1,2))
    return mean_over_entire_dataset/255, std_over_entire_dataset/255;

def get_dataset_stats2(csv_filename):
    files = pd.read_csv(csv_filename)
    mean_over_entire_dataset = 0
    std_over_entire_dataset = 0
    number_of_files = files.shape[0]
    for file in files["ToneMapped"]:
        image = np.array(Image.open(file, "r"),dtype=np.uint16)
        mean_over_entire_dataset = mean_over_entire_dataset + image.mean(axis=(0,1))/number_of_files
        std_over_entire_dataset = std_over_entire_dataset + (image**2).mean(axis=(0,1))/number_of_files

    return mean_over_entire_dataset / 255, np.sqrt(std_over_entire_dataset - np.square(mean_over_entire_dataset))/ 255;

def main():
    mean, std = get_dataset_stats("downloads/image_file_test.csv")
    mean2, std2 = get_dataset_stats2("downloads/image_file_test.csv")
    print(mean)
    print(mean2)
    print(std)
    print(std2)

if __name__ == '__main__':
    main()

