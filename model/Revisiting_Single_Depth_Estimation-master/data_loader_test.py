import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *
import h5py
from ../../../data/convert_distance_to_depth import *


def remap_data(np_array,wanted_max=255):
    return (np_array* wanted_max/np_array.max(0).max(0).astype(np.uint8) )


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform
        

    def __getitem__(self, idx):
        image_name = self.frame.ix[idx, 4]
        depth_name = self.frame.ix[idx, 1]
        # Read file
        if ".hdf5" in image_name:
            image_h5py = h5py.File(image_name, "r")["dataset"][()]
            depth_h5py = h5py.File(depth_name, "r")["dataset"][()]
            image = Image.fromarray(remap_data(image_h5py,255))
            depth = Image.fromarray()
        else:
            image = Image.open(image_name, "r")
            depth = Image.open(depth_name, "r")
            depth = Image.fromarray(convert_distance_to_depth(depth))
        
        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)
    
    
def getTrainingData(batch_size=64):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }

    mean, std = get_image_stats(csv_file="../../data/downloads/image_files.csv")

    transformed_training = depthDataset(csv_file='../../data/downloads/image_files.csv',
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(mean,
                                                      std)
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=4, pin_memory=False)

    return dataloader_training

def getTestingData(batch_size=64):

    mean, std = get_image_stats(csv_file="../../data/downloads/image_files.csv")
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(csv_file="../../data/downloads/image_files.csv",
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [304, 228]),
                                           ToTensor(is_test=True),
                                           Normalize(mean,
                                                     std)
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing