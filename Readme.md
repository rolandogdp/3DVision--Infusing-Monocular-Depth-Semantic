# Semantic-MD: Infusing Monocular Depth With Semantic Signals

### Authors: Alice Mazzoleni and Rolando Grave de Peralta Gonzalez

## Code:

### Starting codebase:

As a starting point, we took the code implementation of the paper [2], which we are using as a base model to improve using semantic information. They have no official licensing file, but they agree to the code being used in projects [cf. this issue](https://github.com/JunjH/Revisiting_Single_Depth_Estimation/issues/29#issuecomment-723793625).

### Improvements/Fixes

The original codebase was made for python 2.7 which is now deprecated. We have updated it and tested it with Python 3.11. It is also compatible with a more recent version of PyTorch 2.0.0.

### Divergences

* Since we are working with a different dataset [2] with different formats and properties, we have updated all the data loading methods and scripts to support our formats (Hdf5) and the dataset differences (notably the presence of NANs).
* We are using only ResNet as a pre-trained backbone because of hardware limitations, so we have only adapted this one.
* We have changed the preprocessing as we are not using as much data augmentation.  You will find the data folder where all scripts were written for this project. The models were highly adapted/created to match our requirements. Notably for the joint training.
* The testing was also modified to be compatible, and we added a demo based on our code. Not all the modifications, additions, and deletions are documented here for obvious readability problems. Checking out the git difference between the initial state of the repo and the last version will provide more insights for those who may want them.

### License:

This code is released as is, without any guarantees. All the code written by us can be used by anyone. As for the base code, you may refer to their GitHub and ask them directly [2].

## Using the code

### Prerequirements

The recommended and tested Python version is 3.11.

To use this code, you will first need to install all the requirements from the requirements.txt file using your favorite package manager. (We would advise creating a virtual environment to avoid potential conflicts.)

#### For a minimal demo run

* You will have to download the pre-trained checkpoints of the model. You will need to place them in ./data/outputs/checkpoints/
* You will need to download the Final_Test_Data.csv and the Final_Test_Data and put them in the ./data/downloads/ folder.


* You will need to set an environment variable pointing to the downloads folder. This path should be absolute. If you are on a Unix based machine, you may go to the downloads folder from a Terminal and then execute:

  ```
  export THREED_VISION_ABSOLUTE_DOWNLOAD_PATH=$(pwd)/
  
  ```
* Then you will be able to go into the ./model/Revisiting_Single_Depth_Estimation-master/ folder and simply run the demo.py file.

### For a train run

* First you will need to download the data by going into the ./data/ folder and running:

  ```
  python3 downloader.py --max_frames 10 --urls 50 --download True
  
  ```
* With the data downloaded you will also need to download the pretrained backbones networks. You only need the ResNet.
* You will need to setup the THREED_VISION_ABSOLUTE_DOWNLOAD_PATH environement variable like in the demo run above.
* After that you should be able to run the model by going into the ./model/Revisiting_Single_Depth_Estimation-master/  folder and running train.py.
* To select the training method you will need to edit the set_method.py and choose the wanted method.
* You may query the train.py script with the -h parameter to get a list of parameters.

An example of a correct call would be:

```
python3 train.py --epochs 30 --learning-rate 0.001 --weight-decay 1e-5 --batch 6
```

### Testing a checkpoint

The recommended way of testing a checkpoint would be using the view-depth.ipynb notebook that has the code used to generate the results we got.You will have to run the following sections: [Needed imports, Specify all the file paths, Loading the model for testing, Testing the model.]

You will need to update the variables to match your specific environnement.

## Citation

TODO: Will be updated with the exact information. In the meanwhile you may use the following:

```
@inproceedings{SemMD2023GraMaz,
title={Semantic-MD: Infusing Monocular Depth With Semantic Signals},
author={Alice Mazzoleni and Rolando Grave de Peralta Gonzalez},
booktitle={},
year={2023}
}
```

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