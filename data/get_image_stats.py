import os
import pandas as pd
import numpy as np
from PIL import Image
import math

import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import itertools
from functools import partial

def split_array_in_number_of_cores(data_list:list, cores_amount:int):
    l = len(data_list)
    step = l //cores_amount +1
    for i in range(0,l,step):
        yield data_list[i:i+step]
        
        
def slices_generator(data_list:list, step:int):
    l = len(data_list)
    for i in range(0,l,step):
        yield data_list[i:i+step]


def get_partial_sums(filenames:list,total_len:int):
    partial_full_sum, partial_full_squared_sum = np.zeros((3,),dtype=np.float64),np.zeros((3,),dtype=np.float64)
    absolute_downloads_path = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH']
    for image_name in filenames:
         
        image = np.array(Image.open(os.path.join(absolute_downloads_path,image_name), "r"),dtype=np.int64)
        partial_full_sum += image.mean(axis=(0,1))/total_len
        partial_full_squared_sum  += (np.square(image)).mean(axis=(0,1))/total_len



    return partial_full_sum, partial_full_squared_sum;


def get_dataset_stats(csv_filename):
    
    files = pd.read_csv(csv_filename)["ToneMapped"]
    full_sum, full_squared_sum = np.zeros((3,)),np.zeros((3,))
    files_len = len(files)
    cpu_counts = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_counts) as pool:
        
        res = pool.map(partial(get_partial_sums, total_len=files_len), split_array_in_number_of_cores(files,cpu_counts))
    for [sum_over_partial_dataset,squared_sum_over_partial_dataset] in res:
        full_sum += sum_over_partial_dataset
        full_squared_sum += squared_sum_over_partial_dataset
    # res = np.array(res)
    # mean_over_entire_dataset = res[:,0,:].mean(axis=0)
    # std_over_entire_dataset = np.sqrt(res[:,1,:].mean(axis=0) - np.square(mean_over_entire_dataset) )
    
    mean_over_entire_dataset1 = full_sum 
    std_over_entire_dataset1 = np.sqrt(full_squared_sum - np.square(mean_over_entire_dataset1) )   
    
    return mean_over_entire_dataset1/255, std_over_entire_dataset1/255;



def get_dataset_stats2(csv_filename):
    files = pd.read_csv(csv_filename)
    mean_over_entire_dataset = 0
    std_over_entire_dataset = 0
    number_of_files = files.shape[0]
    absolute_downloads_path = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH']
    for file in files["ToneMapped"]:
        image = np.array(Image.open(os.path.join(absolute_downloads_path,file), "r"),dtype=np.uint16)
        mean_over_entire_dataset = mean_over_entire_dataset + image.mean(axis=(0,1))/number_of_files
        std_over_entire_dataset = std_over_entire_dataset + (image**2).mean(axis=(0,1))/number_of_files

    return mean_over_entire_dataset / 255, np.sqrt(std_over_entire_dataset - np.square(mean_over_entire_dataset))/ 255;

def main():
    mean, std = get_dataset_stats("./downloads/images_files.csv")
    mean2, std2 = get_dataset_stats2("./downloads/images_files.csv")
    print(mean)
    print(mean2)
    print(std)
    print(std2)

if __name__ == '__main__':
    main()

