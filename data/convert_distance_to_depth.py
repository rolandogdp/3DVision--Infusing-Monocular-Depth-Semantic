import numpy as np
import pandas as pd
from PIL import Image
import os
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
        return self.convert_semantic_label_to_rgb(image)

    def convert_semantic_label_to_rgb(self, image):
        rgb_image = np.empty(shape=(image.size[0], image.size[1], 3), dtype=np.uint8)
        print("SIZE:",image.size)
        mask = (image == -1)
        rgb_image[mask, :] = [255, 255, 255]  # white
        print(f"rgbimage shape bef: {rgb_image.shape}")
        for label in range(1, 40):
            rgb_image[image == label] = self.mappings.iloc[label - 1]
        print(f"rgbimage shape af: {rgb_image.shape}")
        print(f"min:{rgb_image.min()} , max:{rgb_image.max()}")

        return Image.fromarray(rgb_image) 
