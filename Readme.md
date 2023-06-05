# Semantic-MD: Infusing Monocular Depth With Semantic Signals

### Authors:
Alice Mazzoleni & Rolando Grave de Peralta Gonzalez

## Code:

### Starting codebase:
As a starting point we took the code implementation of the paper which we are taking as base model [2] to improve. They have no official Licesing file but they agree to the code being used in projects [cf. this issue](https://github.com/JunjH/Revisiting_Single_Depth_Estimation/issues/29#issuecomment-723793625).

### Improvements/Fixes
The original codebase was made for python 2.7 which is now deprecated. We have updated it and testing with python 3.11. It also is compatible with a more recent version of pytorch 2.0.0.

### Divergences
Since we are working with a different dataset [2] with diffferent formats and properties we have updated all the data loading methods and scripts to support our formats (Hdf5) and the dataset diferences (notably the presence of NANs..).

We are using only ResNet as pretrained backbone because of hardware limitations so we have only adapted this one. 

We have changed the preprocessing as we are not using as much data augmentation. 

You will find the data folder where all scripts were written for this project. 

The models were highly adapted/created to match our requirements. Notably for the joint training.

The testing were also modified to be compatible and added a demo based on our code.


### License:
This code is released as is and without any guarantees. All the code written by us can be used by anyone. As for the base code you may refer to their github and ask them directly [2].

## Using the code
### Prerequirements
The recommended and tested python version is 3.11

To use this code you will first need to install all requirements from the requirements.txt file using your favorite package manager. (We would advice creating a virtual environment to avoid potential conflicts).

#### For a minimal demo run
- You will have to download the pretrained checkpoints of the model. You will need to place them in ./data/outputs/checkpoints/

      
* You will need to download the Final_Test_Data.csv and the Final_Test_Data and put them in the ./data/downloads/ folder. 

* You will need to set an environnement variable pointing to the downloads folder. This path should be absolute. If you are on a Unix based machine you may go to to the downloads folder from a Terminal and then execute:
      
      export THREED_VISION_ABSOLUTE_DOWNLOAD_PATH=$(pwd)/


* Then you will be able to go into the ./model/Revisiting_Single_Depth_Estimation-master/ folder and simply run the demo.py file.


### For a Train run
Todo





## Usefull links:

### Dataset:

* [1] hypersim:
  * https://github.com/apple/ml-hypersim
  * https://github.com/apple/ml-hypersim/tree/main/contrib/mikeroberts3000
  * https://arxiv.org/abs/1803.08673

### Base Model:

 * [2] "Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries" :
    * Codebase Github: https://github.com/JunjH/Revisiting_Single_Depth_Estimation
    * Paper: https://arxiv.org/abs/1803.08673
    * Papers with code: https://paperswithcode.com/paper/revisiting-single-image-depth-estimation
    * Pretrained model: https://drive.google.com/file/d/1QaUkdOiGpMuzMeWCGbey0sT0wXY0xtsj/view