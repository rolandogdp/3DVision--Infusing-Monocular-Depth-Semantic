import os
import pandas as pd
import numpy as np
from PIL import Image
import math

import os
import pandas as pd
import numpy as np
from PIL import Image

def slices_generator(data_list:list, step:int):
    num = 0
    l = len(data_list)
    for i in range(0,l,step):
        # if i >=l:
        #     break
        yield data_list[i:i+step]
        num += 1


def get_partial_sums(filenames:list,total_len:int):
    # print("Did:",len(filenames))
    dataset_array_list = []
    partial_full_sum, partial_full_squared_sum = np.zeros((3,),dtype=np.float64),np.zeros((3,),dtype=np.float64)
    for image_name in filenames:
         
        image = np.array(Image.open(image_name, "r"),dtype=np.int64)
        partial_full_sum += image.mean(axis=(0,1))/total_len
        partial_full_squared_sum  += (np.square(image)).mean(axis=(0,1))/total_len

        # dataset_array_list.append(image)
        # print(np.max(image))
    

    return partial_full_sum, partial_full_squared_sum;


def get_dataset_stats(csv_filename,slice_step=20):
    files = pd.read_csv(csv_filename)["ToneMapped"]
    full_sum, full_squared_sum = np.zeros((3,)),np.zeros((3,))
    files_len = len(files)
    for slice in slices_generator(files,slice_step):
        sum_over_partial_dataset, squared_sum_over_partial_dataset = get_partial_sums(slice,files_len)
        full_sum += sum_over_partial_dataset
        full_squared_sum += squared_sum_over_partial_dataset

    
    mean_over_entire_dataset = full_sum 
    std_over_entire_dataset = np.sqrt(full_squared_sum - np.square(mean_over_entire_dataset) )
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
    mean, std = get_dataset_stats("downloads/image_files.csv")
    mean2, std2 = get_dataset_stats2("downloads/image_files.csv")
    print(mean)
    print(mean2)
    print(std)
    print(std2)

if __name__ == '__main__':
    main()

