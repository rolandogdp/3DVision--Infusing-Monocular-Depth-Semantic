import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
import os
import sys

torch.manual_seed(240)

module_path =  os.path.abspath(os.path.dirname(os.path.realpath(__file__)))+"/../../"
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))+"/../"
if module_path not in sys.path:
    sys.path.append(module_path)

from data.get_image_stats import *
from data.convert_distance_to_depth import *
from set_method import *
from nyu_transform import *
import h5py

import set_method

def remap_data(np_array,wanted_max=255):
    return (np_array/np_array.max(axis=(0,1))*wanted_max).astype(np.uint8)

class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform_rgb_image=None, transform_depth_image=None, transform_segmentation_mask=None):
        self.frame = pd.read_csv(csv_file)
        self.transform_rgb = transform_rgb_image
        self.transform_depth = transform_depth_image
        self.transform_segmentation_mask = transform_segmentation_mask
        self.absolute_downloads_path = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH']
        if(MyMethod.my_method == Method.SEGMENTATIONMASKGRAYSCALE):
            self.grayscale_conversion = Grayscale()
            self.convert_semantic_label_to_rgb = ConvertSemanticLabelsToRGB()
        if(MyMethod.my_method == Method.SEGMENTATIONMASKBOUNDARIES):
            self.canny_edge_detection = CannyEdgeDetection(0.01)
            self.convert_semantic_label_to_rgb = ConvertSemanticLabelsToRGB()

    def __getitem__(self, idx):
        image_name = self.absolute_downloads_path + self.frame["ToneMapped"][idx]
        depth_name = self.absolute_downloads_path + self.frame["Depth"][idx]
        if(MyMethod.my_method != Method.NOSEGMENTATIONCUES):
            segmentation_mask_name = self.absolute_downloads_path + self.frame["Segmentation"][idx]

        # Read file
        if ".hdf5" in image_name:
            image_h5py = h5py.File(image_name, "r")["dataset"][()]
            image = Image.fromarray(image_h5py)
        else:
            image = Image.open(image_name, "r")
            if (np.isnan(image).any()):
                print("are there already nan values in raw rgb data: ", np.isnan(image).any());

        if self.transform_rgb:
            #sample = self.transform(sample)
            image = self.transform_rgb(image)

        if ".hdf5" in depth_name:
            depth_h5py = h5py.File(depth_name, "r")["dataset"][()]
            depth = Image.fromarray(convert_distance_to_depth(depth_h5py))
        else:
            depth = Image.open(depth_name, "r")
            depth = Image.fromarray(convert_distance_to_depth(depth))

        if self.transform_depth:
            depth = self.transform_depth(depth)

        if(MyMethod.my_method != Method.NOSEGMENTATIONCUES):
            if ".hdf5" in segmentation_mask_name:
                segmentation_mask_h5py = h5py.File(segmentation_mask_name, "r")["dataset"][()]
                segmentation_mask = Image.fromarray(segmentation_mask_h5py)
            else:
                segmentation_mask = Image.open(segmentation_mask_name, "r")

            #binary image also one hod encoded vector??
            #grayscale image normalized with mean and std from std images or segmentation masks?
        if(MyMethod.my_method == Method.SEGMENTATIONMASKGRAYSCALE and self.transform_segmentation_mask):
            segmentation_mask = self.convert_semantic_label_to_rgb(segmentation_mask)
            segmentation_mask = self.grayscale_conversion(segmentation_mask)
            segmentation_mask = self.transform_segmentation_mask(segmentation_mask)
            print("am i in here")
        elif(MyMethod.my_method == Method.SEGMENTATIONMASKONEHOT and self.transform_segmentation_mask):
            for transform in self.transform_segmentation_mask[0:-1]: #dont normalize
                segmentation_mask = transform(segmentation_mask)
            #convert to one-hot-encoded vector
            segmentation_mask_one_hot_encoded = torch.concat([segmentation_mask[segmentation_mask == label] for label in range(1,40)], axis=0)
            segmentation_mask = segmentation_mask_one_hot_encoded
        elif(MyMethod.my_method == Method.SEGMENTATIONMASKBOUNDARIES):
            segmentation_mask = self.convert_semantic_label_to_rgb(segmentation_mask)
            segmentation_mask = self.cannyedgedetection(segmentation_mask)
            if(self.transform_segmentation_mask):
                for transform in self.transform_segmentation_mask[0:-1]: #dont normalize
                    segmentation_mask = transform(segmentation_mask)
            segmentation_mask = torch.concat([segmentation_mask[segmentation_mask == 0].int, segmentation_mask[segmentation_mask == 1].int], axis=0)

        if(MyMethod.my_method != Method.NOSEGMENTATIONCUES):
            print("am i in here")
            image = torch.concat([segmentation_mask, image], axis=0)

        print(image.shape)
        sample = {'image': image, 'depth': depth}

        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(batch_size=64, csv_filename="image_files.csv"):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }

    filename = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH']  + csv_filename

    mean, std = get_dataset_stats(csv_filename=filename)  # TODO: only extracts image stats of particular subset but not of the entire dataset

    if(MyMethod.my_method != Method.NOSEGMENTATIONCUES):
        transform_segmentation_mask = transforms.Compose([Scale(240, Image.NEAREST), RandomHorizontalFlip(), RandomRotate(5), CenterCrop([304,228], [304, 228]), ToTensor(), Normalize(mean, std)])
    else:
        transform_segmentation_mask = None

    print(f"TRAINING DATA Mean and std : mean:{mean} ,std:{std}")

    transformed_training = depthDataset(csv_file=filename,
                                        transform_rgb_image=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [304, 228]),
                                            ToTensor(),
                                            Normalize(mean,
                                                      std)
                                        ]), transform_depth_image=transforms.Compose([Scale(240, Image.NEAREST),
                                                                                      RandomHorizontalFlip(),
                                                                                      RandomRotate(5),
                                                                                      CenterCrop([304,228],
                                                                                                 [152, 114]), ToTensor()]), transform_segmentation_mask=transform_segmentation_mask)

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=5, pin_memory=False)

    return dataloader_training


def getTestingData(batch_size=64, csv_filename="images_files.csv"):

    """
    image stats for nyu2 dataset
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    """

    filename = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] + csv_filename

    # print(filename)

    mean, std = get_dataset_stats(csv_filename=filename) #TODO: only extracts image stats of particular subset but not of the entire dataset

    if (MyMethod.my_method != Method.NOSEGMENTATIONCUES):
        transform_segmentation_mask = transforms.Compose(
            [Scale(240, Image.NEAREST), RandomHorizontalFlip(), RandomRotate(5), CenterCrop([304, 228], [304, 228]),
             ToTensor(), Normalize(mean, std)])
    else:
        transform_segmentation_mask = None

    transformed_testing = depthDataset(csv_file=filename,
                                        transform_rgb_image=transforms.Compose([
                                            Scale(240),
                                            CenterCrop([304, 228], [304, 228]),
                                            ToTensor(is_test=True),
                                            Normalize(mean,std)
                                        ]), transform_depth_image=transforms.Compose([Scale(240, Image.NEAREST),
                                                                                      RandomHorizontalFlip(),
                                                                                      RandomRotate(5),
                                                                                      CenterCrop([304,228],
                                                                                                 [152, 114]), ToTensor()]), transform_segmentation_mask=transform_segmentation_mask)
    
    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=True, num_workers=0, pin_memory=False)

    return dataloader_testing


def getValidationData(batch_size=64, csv_filename="image_files.csv"):
    filename = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH']  + csv_filename


    if(MyMethod.my_method != Method.NOSEGMENTATIONCUES):
        transform_segmentation_mask = transforms.Compose([Scale(240, Image.NEAREST), RandomHorizontalFlip(), RandomRotate(5), CenterCrop([304,228], [304, 228]), ToTensor()])
    else:
        transform_segmentation_mask = None


    transformed_training = depthDataset(csv_file=filename,
                                        transform_rgb_image=transforms.Compose([
                                            Scale(240),
                                            CenterCrop([304, 228], [304, 228]),
                                            ToTensor(is_test=True),
                                            # Normalize(mean,
                                            #           std)
                                        ]), transform_depth_image=transforms.Compose([Scale(240, Image.NEAREST),
                                                                                      CenterCrop([304,228],
                                                                                                 [152, 114]), ToTensor(is_test=True)]), transform_segmentation_mask=transform_segmentation_mask)

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=5, pin_memory=False)

    return dataloader_training