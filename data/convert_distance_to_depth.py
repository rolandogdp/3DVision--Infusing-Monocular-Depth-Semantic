import numpy as np
import pandas as pd
import torch.nn
from PIL import Image
import os
import sys

module_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))+"/../"
if module_path not in sys.path:
    sys.path.append(module_path)

from nyu_transform import _is_pil_image, _is_numpy_image
#Arg: pass depth image as numpy array
#return: returns depth values as numpy array
def convert_distance_to_depth(np_depth_array):
    intHeight, intWidth = np_depth_array.shape
    fltFocal = 886.81
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    return np_depth_array / np.linalg.norm(npyImageplane, 2, 2) * fltFocal


class ConvertSemanticLabelsToRGB(object):

    def __init__(self):
        path = os.path.abspath(os.path.dirname(__file__))
        self.mappings = pd.read_csv(path+"/semantic_label_descs.csv",usecols=["semantic_color_r", "semantic_color_g", "semantic_color_b"])

    def __call__(self, image):

        if _is_pil_image(image):
            return self.convert_semantic_label_to_rgb(image)
        elif _is_numpy_image(image):
            return self.convert_semantic_label_to_rgb(image)
        else:
            return self.convert_semantic_label_to_rgb_tensor(image)


    def convert_semantic_label_to_rgb(self, image):
        rgb_image = np.empty(shape=(image.size[1], image.size[0], 3), dtype=np.uint8)
        image = np.asarray(image)
        rgb_image[image==-1, :] = [255, 255, 255]  # white
        for label in range(1, 40):
            rgb_image[image==label, :] = self.mappings.iloc[label - 1]
        return Image.fromarray(rgb_image)

    def convert_semantic_label_to_rgb_tensor(self, image):
        rgb_image = torch.empty(size=(3, image.shape[0], image.shape[1]), dtype=torch.uint8)
        rgb_image[:, image == -1] = torch.tensor([[255], [255], [255]], dtype=torch.uint8)  # white
        for label in range(1, 40):
            rgb_image[:, image == label] = torch.tensor(self.mappings.iloc[label - 1], dtype=torch.uint8).unsqueeze(-1)
        return rgb_image

class ConvertProbabilitiesToLabels(object):

    def __init__(self, segmentation_classes_csv="all_classes.csv"):
        path = os.path.abspath(os.path.dirname(__file__))
        self.mappings = pd.read_csv(path+"/semantic_label_descs.csv", usecols=["semantic_color_r", "semantic_color_g", "semantic_color_b"])
        self.softmax = torch.nn.Softmax(dim=1)
        self.segmentation_classes = pd.read_csv(path+"/segmentation_classes/"+segmentation_classes_csv, usecols=["semantic_id"])

    def __call__(self, segmentation_pediction):
        return self.convert_probability_to_labels(segmentation_pediction)

    def convert_probability_to_labels(self, segmentation_prediction):
        segmentation_probabilities = self.softmax(segmentation_prediction)
        segmentation_labels = torch.argmax(segmentation_probabilities, dim=1)
        class_labels = self.segmentation_classes.iloc[segmentation_labels.reshape(-1)].values.flatten().reshape(
            segmentation_labels.shape)
        return torch.from_numpy(class_labels)




