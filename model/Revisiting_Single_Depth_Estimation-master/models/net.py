from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
from . import modules
from torchvision import utils

from . import senet
from . import resnet
from . import densenet

class joint_model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(joint_model, self).__init__()

        self.E = Encoder
        self.D_depth = modules.D(num_features)
        self.D_segmentation = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R_depth = modules.R(block_channel)
        self.R_segmentation = modules.R_classification(block_channel)


    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder_depth = self.D_depth(x_block1, x_block2, x_block3, x_block4)
        x_decoder_segmentation = self.D_segmentation(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder_depth.size(2),x_decoder_depth.size(3)])
        out_depth = self.R_depth(torch.cat((x_decoder_depth, x_mff), 1))
        out_segmentation = self.R_segmentation(torch.cat((x_decoder_segmentation, x_mff), 1))

        return out_depth, out_segmentation

class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(block_channel)


    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out
