import argparse
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
# plt.set_cmap("jet")

import os
import sys

from set_method import my_method, Method


module_path = os.path.abspath(os.path.dirname(os.getcwd()))+"/../" # Only works if cwd correctly gets the model folder.

test_path = os.path.abspath(os.path.dirname(os.getcwd()))

if test_path not in sys.path:
    sys.path.append(test_path)

if module_path not in sys.path:
    sys.path.append(module_path)

from data.convert_distance_to_depth import *
from loaddata import *

from data.plot_scripts import * 

from set_method import *

from test import edge_detection

import warnings

ABSOLUTE_DOWNLOAD_PATH = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH']
if(torch.cuda.is_available()):
    device = "cuda:0"
else:
    device="cpu"
        

def define_model(is_resnet, is_densenet, is_senet, num_segmentation_classes, pretrained = False):
    if is_resnet:
        original_model = resnet.resnet50(num_segmentation_classes=num_segmentation_classes, pretrained=pretrained)
        Encoder = modules.E_resnet(original_model)
        if my_method is Method.JOINTLEARNING:
            model = net.joint_model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
        else:
            model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=pretrained)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def test_sample(model, test_loader): 
    model.eval()
    depth_results = []
    if(torch.cuda.is_available()):
        model.to("cuda:0")
    else:model.to("cpu")
    for i, sample_batched in enumerate(test_loader):
        torch.cuda.empty_cache()
        
        image, depth = sample_batched['image'], sample_batched['depth']
        if(torch.cuda.is_available()):
            depth = depth.cuda(non_blocking=True) #
            image = image.cuda()
        image = torch.autograd.Variable(image, requires_grad=False)
        depth = torch.autograd.Variable(depth, requires_grad=False)
        output = model(image)
        output = torch.nn.functional.interpolate(output, size=[depth.size(2),depth.size(3)], mode='bilinear')
        #display_tensor_data(output[0,:].detach())
        #display_tensor_data(depth[0,:])
        #plt.imshow(output[0,:].permute(1, 2, 0).detach().numpy())
        #plt.show()
        depth_results.append([output.detach(),depth.detach() ,image.detach()])
        torch.cuda.empty_cache()
        
    return depth_results


def display_tensor_data_many(tensors_dict:dict, remap, titles, colorbar, figsize=(15, 15), fontsize=12, plot_name=None):
    
    n_columns = len(titles)
    n_rows = len(tensors_dict[list(tensors_dict.keys())[0]])
    # print(n_rows)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize)
    #fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.13, hspace=0.0)
    
    for key_index, key in enumerate(tensors_dict.keys()):
        for tensor_index, tensor in enumerate(tensors_dict[key]): 
            if remap[key] is not None:
                tensor = remap[key](tensor)     
            pcm = axes[key_index].imshow(tensor.permute(1, 2, 0))
            axes[key_index].set_xticks([])
            axes[key_index].set_yticks([])
            
            if colorbar[key] == True:
                fig.colorbar(pcm, ax=axes[key_index], fraction=0.035, pad=0.02) #0.046, 0.04
            
            
        axes[key_index].set_title(titles[key], fontsize=fontsize)
        
    fig.tight_layout()
    plt.show()



def main():
    
    validation_image_path_csv = "Final_Test_Data.csv"
    
    # selected_segmentation_classes = "only_relevant_classes.csv"
    selected_segmentation_classes = "no_chairs.csv"
    
    choosen_model = "checkpointapple-.06-02-2023-20-44-56-Method.SEGMENTATIONMASKONEHOT-no_chairs.csv.final.pth.tar"

    data_path = os.path.abspath(os.path.dirname("./../../data/"))
    csv_file_reader = open(data_path+"/segmentation_classes/" + selected_segmentation_classes)
    num_segmentation_classes = sum(1 for line in csv_file_reader) - 1
    
    
        
    model = define_model(is_resnet=True, is_densenet=False, is_senet=False,num_segmentation_classes=num_segmentation_classes,pretrained=False )
    
    # choosen_model = "checkpointapple-.06-02-2023-20-54-43-Method.SEGMENTATIONMASKONEHOT-only_relevant_classes.csv.final.pth.tar"
    model_checkpoint_path = ABSOLUTE_DOWNLOAD_PATH+"../outputs/checkpoints/"+choosen_model
    
        
    state_dict = torch.load(model_checkpoint_path, map_location=torch.device(device))["state_dict"]
    model.load_state_dict(state_dict=state_dict)
    

    model.eval()
    
    


    test_loader = getTestingData(1, validation_image_path_csv, selected_segmentation_classes)
  
    test(model, test_loader)
    
    
    


def test( model, test_loader):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        depth_results = test_sample(model, test_loader)
    
    plotname = None
    for i, tensors in enumerate(depth_results):
        outputs,depths,rgbs = tensors
        outputs = outputs[0,:,:,:].cpu()
        depths = depths[0,:,:,:].cpu()
        rgbs = rgbs[0,-3:,:,:].cpu()
        if(plotname != None):
            final_plotname = plotname + "_image_"+str(i)
        else: 
            final_plotname = None
        tensor_dict = {"RGB":[rgbs],"depths":[depths],"outputs":[outputs]}
        remap_dict = {"RGB":denormalize,"depths":None,"outputs":None}
        names_dict = {"RGB":"RGB","depths":"GT","outputs":"Prediction"}
        color_bar_dict = {"RGB":False,"depths":True,"outputs":True}
        display_tensor_data_many(tensors_dict=tensor_dict,remap= remap_dict, titles=names_dict,colorbar=color_bar_dict)
    
if __name__ == '__main__':
    main()
