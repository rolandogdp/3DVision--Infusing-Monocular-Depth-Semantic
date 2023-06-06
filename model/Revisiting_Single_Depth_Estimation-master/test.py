import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import sys
import os
this_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
module_path =  this_path+"/../../"
os.chdir(this_path)
if module_path not in sys.path:
    sys.path.append(module_path)
# print("HERE1:",sys.path)
from models import modules, net, resnet, densenet, senet
import loaddata
import util
import numpy as np
import sobel

import csv

import datetime

from train import define_model

from set_method import my_method

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

parser = argparse.ArgumentParser(description='PyTorch DenseNet Testing')

parser.add_argument('--batch', default=1, type=int,
                    help='sets the batch size for testing')

parser.add_argument('--selected_segmentation_classes', default="all_classes.csv", type=str, help="For the method Joint Learning and OneHotencoded Vector one can pass a file with the respective segmentation classes, it should be the same file used for the training")

parser.add_argument('--pretrained_model', default="./pretrained_model/model_senet", type=str,
                    help='tar file name of pretrained model')

def main():
    global args
    args = parser.parse_args()

    data_path = os.path.abspath(os.path.dirname("./../../data/"))
    csv_file_reader = open(data_path + "/segmentation_classes/" + args.selected_segmentation_classes)
    num_segmentation_classes = sum(1 for line in csv_file_reader) - 1
    model = define_model(is_resnet=True, is_densenet=False, is_senet=False, num_segmentation_classes=num_segmentation_classes, pretrained=False)

    pretrained_model_path = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] +'../outputs/checkpoints/'
    pretrained_model = args.pretrained_model

    state_dict = torch.load(pretrained_model_path + pretrained_model, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)

    model = torch.nn.DataParallel(model).to(device)

    batch_size = args.batch
    
    test_loader = loaddata.getTestingData(batch_size, "test_data.csv", args.selected_segmentation_classes)
    test(test_loader, model, 1.5)

def test(test_loader, model, thre):
    model.eval()

    totalNumber = len(test_loader)
    tresholds = [0.25, 0.5, 1.0, 1.25]
    tresholds_res = [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ]

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    for i, sample_batched in enumerate(test_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        if(torch.cuda.is_available()):
            depth = depth.cuda(non_blocking=True) #
            image = image.cuda()

        image = torch.autograd.Variable(image, requires_grad=False)
        depth = torch.autograd.Variable(depth, requires_grad=False)
 
        output = model(image)
        output = torch.nn.functional.interpolate(output, size=[depth.size(2),depth.size(3)], mode='bilinear')

        _output, _depth, num_non_nans = util.setNanToZero(output, depth)

        depth_edge = edge_detection(_depth,1)
        output_edge = edge_detection(_output,1)

        batchSize = depth.size(0)
        errors = util.evaluateError(_output, _depth, num_non_nans)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        for tresh_index,res in enumerate(tresholds_res):
            thre = tresholds[tresh_index]
            edge1_valid = (depth_edge > thre)
            edge2_valid = (output_edge > thre)

            # print("max GT edge map: ", edge1_valid.max())
            # print("min GT edge map: ", edge1_valid.min())
            # print("max pred edge map: ", edge2_valid.max())
            # print("min pred edge map: ", edge2_valid.min())

            nvalid = np.sum(torch.ne(edge1_valid, edge2_valid).float().data.cpu().numpy())
            A = nvalid / num_non_nans#(depth.size(2)*depth.size(3)) #how many pixel are the same in edge map in percentage

            nvalid2 = np.sum(torch.logical_and(edge1_valid, edge2_valid).float().data.cpu().numpy()) #number of true positive

            P = nvalid2/(np.sum(edge2_valid.data.cpu().numpy())) #precision
            R = nvalid2/(np.sum(edge1_valid.data.cpu().numpy())) #recall

            epsilon = 10**(-10)
            F = (2 * P * R) / (P + R + epsilon) #precision and recall?
            res[0] += A
            res[1] += P
            res[2] += R
            res[3] += F

            # Ae += A
            # Pe += P
            # Re += R
            # Fe += F
    segmentationError_all = {}
    for tresh_index, res in enumerate(tresholds_res):
        Av = (res[0] / totalNumber).item()
        Pv = res[1] / totalNumber
        Rv = res[2] / totalNumber
        Fv = res[3] / totalNumber
        # print(Av)

        segmentationError = {f'Precision_of_EdgeMap-{tresholds[tresh_index]}': Pv, f'Recall_of_EdgeMap-{tresholds[tresh_index]}': Rv, f'F_Measure-{tresholds[tresh_index]}': Fv, f'Relative_EdgeMap_Error-{tresholds[tresh_index]}': Av}
        segmentationError_all.update(segmentationError)

    averageError = util.averageErrors(errorSum, totalNumber)
    averageError['RMSE'] = np.sqrt(averageError['MSE'])

    print(averageError)
    print(segmentationError_all)

    now = datetime.datetime.now()
    filename_date = f".{str(now.strftime('%m-%d-%Y-%H-%M-%S'))}"
    filename_error_norms = f"test_-{filename_date}-{my_method}_error_norms.csv"
    path = f"{os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] + '../outputs/results/test_error_norms/'}"
    keys = {'MSE', 'RMSE', 'ABS_REL', 'LG10',
                    'MAE', 'DELTA1', 'DELTA2', 'DELTA3', 'Precision_of_EdgeMap', 'Recall_of_EdgeMap', 'F_Measure', 'Relative_EdgeMap_Error'}

    averageError.update(segmentationError_all)
    print(type(averageError))
    print(averageError)

    with open(path + filename_error_norms, mode="w", newline='') as file:
        w = csv.DictWriter(file, averageError.keys())
        w.writeheader()
        w.writerow(averageError)


def test_sample_joint(test_loader, model, thre): 
    model.eval()
    totalNumber = len(test_loader)
    tresholds = [0.25, 0.5, 1.0, 1.25]
    tresholds_res = [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ]

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0
    

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    
    depth_results = []
    sgmentation_results = []
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
        depth_output, segmentation_output = model(image)
        depth_output = torch.nn.functional.interpolate(depth_output, size=[depth.size(2),depth.size(3)], mode='bilinear')

        _output, _depth, num_non_nans = util.setNanToZero(depth_output, depth)
        


        depth_edge = edge_detection(_depth,1)
        output_edge = edge_detection(_output,1)

        batchSize = depth.size(0)
        errors = util.evaluateError(_output, _depth, num_non_nans)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        for tresh_index,res in enumerate(tresholds_res):
            thre = tresholds[tresh_index]
            edge1_valid = (depth_edge > thre)
            edge2_valid = (output_edge > thre)

            nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
            A = nvalid / num_non_nans#(depth.size(2)*depth.size(3)) #how many pixel are the same in edge map in percentage

            nvalid2 = np.sum(((edge1_valid + edge2_valid) ==2).float().data.cpu().numpy()) #number of true positive
            P = nvalid2/(np.sum(edge2_valid.data.cpu().numpy())) #precision
            R = nvalid2/(np.sum(edge1_valid.data.cpu().numpy())) #recall

            epsilon = 10**(-10)
            F = (2 * P * R) / (P + R + epsilon) #precision and recall?

            res[0] += A
            res[1] += P
            res[2] += R
            res[3] += F

            # Ae += A
            # Pe += P
            # Re += R
            # Fe += F
    segmentationError_all = {}
    for tresh_index, res in enumerate(tresholds_res):
        Av = (res[0] / totalNumber).item()
        Pv = res[1] / totalNumber
        Rv = res[2] / totalNumber
        Fv = res[3] / totalNumber
        # print(Av)

        segmentationError = {f'Precision_of_EdgeMap-{tresholds[tresh_index]}': Pv, f'Recall_of_EdgeMap-{tresholds[tresh_index]}': Rv, f'F_Measure-{tresholds[tresh_index]}': Fv, f'Relative_EdgeMap_Error-{tresholds[tresh_index]}': Av}
        segmentationError_all.update(segmentationError)
    averageError = util.averageErrors(errorSum, totalNumber)
    averageError['RMSE'] = np.sqrt(averageError['MSE'])

    print(averageError)
    print(segmentationError_all)

    now = datetime.datetime.now()
    filename_date = f".{str(now.strftime('%m-%d-%Y-%H-%M-%S'))}"
    filename_error_norms = f"test_-{filename_date}-{my_method}_error_norms.csv"
    path = f"{os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] + '../outputs/results/test_error_norms/'}"
    keys = {'MSE', 'RMSE', 'ABS_REL', 'LG10',
                    'MAE', 'DELTA1', 'DELTA2', 'DELTA3', 'Precision_of_EdgeMap', 'Recall_of_EdgeMap', 'F_Measure', 'Relative_EdgeMap_Error'}

    averageError.update(segmentationError_all)
    print(type(averageError))
    print(averageError)

    with open(path + filename_error_norms, mode="w", newline='') as file:
        w = csv.DictWriter(file, averageError.keys())
        w.writeheader()
        w.writerow(averageError)

        
    return depth_results,sgmentation_results  

def edge_detection(depth,channel_inputs=1):
    get_edge = sobel.Sobel(channel_inputs).to(device)

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel

if __name__ == '__main__':
    main()


