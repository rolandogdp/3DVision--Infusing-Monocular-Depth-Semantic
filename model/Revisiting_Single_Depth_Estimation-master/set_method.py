"""
Specify the format of the segmentation maps to be used as an additional input channel
"""

from enum import Enum
import sys
import os
module_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
if module_path not in sys.path:
    sys.path.append(module_path)

class Method(Enum):
    NOSEGMENTATIONCUES = 0
    SEGMENTATIONMASKGRAYSCALE = 1
    SEGMENTATIONMASKBOUNDARIES = 2
    SEGMENTATIONMASKONEHOT = 3

"""
class MyMethod:
    my_method = Method.NOSEGMENTATIONCUES
    @staticmethod
    def set_method(method_index):
        MyMethod.my_method = Method(method_index)
        
"""

global my_method
my_method = Method.SEGMENTATIONMASKBOUNDARIES

