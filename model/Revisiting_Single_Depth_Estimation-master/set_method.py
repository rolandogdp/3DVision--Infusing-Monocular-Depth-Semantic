"""
Specify the format of the segmentation maps to be used as an additional input channel
"""

from enum import Enum

class Method(Enum):
    NOSEGMENTATIONCUES = 0
    SEGMENTATIONMASKGRAYSCALE = 1
    SEGMENTATIONMASKBOUNDARIES = 2
    SEGMENTATIONMASKONEHOT = 3

def init(method):
    global my_method
    my_method = method


