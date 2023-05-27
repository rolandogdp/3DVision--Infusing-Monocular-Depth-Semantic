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

parser.add_argument('--pretrained_model', default="./pretrained_model/model_senet", type=str,
                    help='tar file name of pretrained model')

def main():
    global args
    args = parser.parse_args()

    model = define_model(is_resnet=True, is_densenet=False, is_senet=False, pretrained=False)

    pretrained_model_path = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] +'../outputs/checkpoints/'
    pretrained_model = args.pretrained_model

    state_dict = torch.load(pretrained_model_path + pretrained_model, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)

    model = torch.nn.DataParallel(model).to(device)

    batch_size = args.batch
    
    test_loader = loaddata.getTestingData(batch_size, "test_data.csv")
    test(test_loader, model, 2e-04)

def test(test_loader, model, thre):
    model.eval()

    totalNumber = len(test_loader)

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

        edge1_valid = (depth_edge > thre)
        edge2_valid = (output_edge > thre)

        nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
        A = nvalid / num_non_nans#(depth.size(2)*depth.size(3)) #how many pixel are the same in edge map in percentage

        nvalid2 = np.sum(((edge1_valid + edge2_valid) ==2).float().data.cpu().numpy()) #number of true positive
        P = nvalid2/(np.sum(edge2_valid.data.cpu().numpy())) #precision
        R = nvalid2/(np.sum(edge1_valid.data.cpu().numpy())) #recall

        epsilon = 10**(-10)
        F = (2 * P * R) / (P + R + epsilon) #precision and recall?

        Ae += A
        Pe += P
        Re += R
        Fe += F

    Av = (Ae / totalNumber).item()
    Pv = Pe / totalNumber
    Rv = Re / totalNumber
    Fv = Fe / totalNumber
    print(Av)

    segmentationError = {'Precision_of_EdgeMap': Pv, 'Recall_of_EdgeMap': Rv, 'F_Measure': Fv, 'Relative_EdgeMap_Error': Av}
    averageError = util.averageErrors(errorSum, totalNumber)
    averageError['RMSE'] = np.sqrt(averageError['MSE'])

    print(averageError)
    print(segmentationError)

    now = datetime.datetime.now()
    filename_date = f".{str(now.strftime('%m-%d-%Y-%H-%M-%S'))}"
    filename_error_norms = f"test_-{filename_date}-{my_method}_error_norms.csv"
    path = f"{os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] + '../outputs/results/test_error_norms/'}"
    keys = {'MSE', 'RMSE', 'ABS_REL', 'LG10',
                    'MAE', 'DELTA1', 'DELTA2', 'DELTA3', 'Precision_of_EdgeMap', 'Recall_of_EdgeMap', 'F_Measure', 'Relative_EdgeMap_Error'}

    averageError.update(segmentationError)
    print(type(averageError))
    print(averageError)

    with open(path + filename_error_norms, mode="w", newline='') as file:
        w = csv.DictWriter(file, averageError.keys())
        w.writeheader()
        w.writerow(averageError)



   

def edge_detection(depth,channel_inputs=1):
    get_edge = sobel.Sobel(channel_inputs).to(device) #.cuda()

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
        torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel

if __name__ == '__main__':
    main()


