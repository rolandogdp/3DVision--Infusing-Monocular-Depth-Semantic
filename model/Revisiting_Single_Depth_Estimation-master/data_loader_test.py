import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *
import h5py

def remap_data(np_array,wanted_max=255):
    return (np_array* wanted_max/np_array.max(0).max(0).astype(np.uint8) )


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform
        

    def __getitem__(self, idx):
        image_name = self.frame.ix[idx, 0]
        depth_name = self.frame.ix[idx, 1]
        # Read file
        if ".hdf5" in image_name:
            image_h5py = h5py.File(image_name, "r")["dataset"][()]
            depth_h5py = h5py.File(depth_name, "r")["dataset"][()]
            #Image.fromarray(remap_data(rgbdata,255))
            image = Image.fromarray(remap_data(image_h5py))
            depth = Image.fromarray(remap_data(depth_h5py ))
        else:
            image = Image.open(image_name)
            depth = Image.open(depth_name)
        
        
        # 
        

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
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

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
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=4, pin_memory=False)

    return dataloader_training
