import argparse

import csv
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
import util
import numpy as np
import sobel
import datetime
import os
from models import modules, net, resnet, densenet, senet
from enum import Enum
from set_method import my_method, Method

torch.cuda.seed()

csv_header_created = (False,False,False)

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=1, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')

parser.add_argument('--batch', default=2, type=int,
                    help='sets the batch size for training')
#parser.add_argument('--method', default=0, type=int, help="specify in which format the segmentation maps should be used as an additional input, options NOSEGMENTATIONCUES=0,  SEGMENTATIONMASKGRAYSCALE=1, SEGMENTATIONMASKBOUNDARIES=2, SEGMENTATIONMASKONEHOT=3")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def define_model(is_resnet, is_densenet, is_senet, pretrained = True):
    if is_resnet:
        original_model = resnet.resnet50(pretrained=pretrained)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=pretrained)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


def main():
    # print("GPU VRAM MAIN:",torch.cuda.mem_get_info())
    global args
    args = parser.parse_args()
    #set_method.mymethod(args.method) #initialize the desired method
    print("What is the value of my_method after initializing: ", my_method)
    model = define_model(is_resnet=True, is_densenet=False, is_senet=False, pretrained=True)
    # print("GPU VRAM model defined:",torch.cuda.mem_get_info())
    now = datetime.datetime.now()
    filename_date = f".{str(now.strftime('%m-%d-%Y-%H-%M-%S'))}"

    if not torch.cuda.is_available():
        model.cpu()
        batch_size = 2
        print("GPU NOT DETECTED, RUNNING ON CPU")
    else:
        print("CUDA DETECTED, RUNNING ON GPU !")
        model = model.cuda()
        # batch_size = 4
        batch_size = args.batch
    # print("GPU VRAM ifs:",torch.cuda.mem_get_info())

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # print("GPU VRAM optimizer:",torch.cuda.mem_get_info())
    print("Starting to load the data.")
    train_loader = loaddata.getTrainingData(batch_size,"train_data.csv")
    first_batch_of_train_loader = next(iter(train_loader))

    validation_loader = loaddata.getValidationData(1,"validation_data.csv")
    first_batch_of_validation_loader = next(iter(validation_loader))
    print("DataLoader finished loading")
    
    training_depth_res = []
    validation_depth_res = []
    filename_train = f"train-{filename_date}-{my_method}"
    filename_val = f"validation-{filename_date}-{my_method}"
    keys = ["loss_depth","loss_dx","loss_dy","loss_normal","loss" ]
    try:
        p = f"{os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] +'../outputs/results/'}"
        with open(p+filename_train+"-results.csv",mode="w", newline='') as file:
            w = csv.DictWriter(file, keys)
            w.writeheader()
        
        with open(p+filename_val+"-results.csv",mode="w", newline='') as file:
            w = csv.DictWriter(file, keys)
            w.writeheader()
    except Exception as e:
        print("Exception while trying to write headers, got:",str(e))

    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        start_time_loop = time.time()
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch)
        end_time_loop = time.time()
        print(f"EPOCH TRAINED FOR :{end_time_loop-start_time_loop} ")
        res_training = validation(batch=first_batch_of_train_loader,model=model)
        training_depth_res.append(res_training["output"])
        res_training.pop("output")
        save_results(res_training,filename_train)

        print("Saved training results")
        if epoch % 5 == 0:
            res_validation = validation(batch=first_batch_of_validation_loader,model=model)
            validation_depth_res.append(res_validation["output"])
            res_validation.pop("output")
            
            save_results(res_validation,filename_val)
            
            print("Saved validation data.")
        if epoch % 1 == 0:
            file = f"{os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] +'../outputs/checkpoints/'}checkpointapple-{filename_date}-{epoch}--{my_method}.pth.tar"

            print("Saving checkpoint to:", file)
            save_checkpoint({'state_dict': model.state_dict()}, file)

    path = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] +"../outputs/results/"

    # Write the outputs
    file=path+filename_train+"-depth.pt"
    torch.save(torch.concat(training_depth_res).unsqueeze(1),file)
    file=path+filename_val+"-depth.pt"
    torch.save(torch.concat(validation_depth_res).unsqueeze(1),file)

    end_time = time.time()
    print(f"TRAINED FOR:{end_time-start_time} ")

    file = f"{os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] +'../outputs/checkpoints/'}checkpointapple-{filename_date}-{my_method}-final.pth.tar"
    print("Saving checkpoint to:", file)
    save_checkpoint({'state_dict': model.state_dict()}, file)


def train(train_loader, model, optimizer, epoch):
    # if(torch.cuda.is_available()):
    #     print("GPU VRAM 1:",torch.cuda.mem_get_info())
    print(f"===== EPOCH:{epoch} ====== ")
    criterion = nn.L1Loss()
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    # print("GPU VRAM before Sobel:",torch.cuda.mem_get_info())

    if(torch.cuda.is_available()):
        get_gradient = sobel.Sobel(1).cuda()
    else:
        get_gradient = sobel.Sobel(1).cpu()

    if(torch.cuda.is_available()):
        print("GPU VRAM After Sobel:",torch.cuda.mem_get_info())

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        # print("DOING ITERATION:",i)
        # print("GPU VRAM:",torch.cuda.mem_get_info())
        image, depth = sample_batched['image'], sample_batched['depth']
        if depth.isnan().any():
            print("="*30,"NAN IN INITIAL DEPTH")
        # print(f"depth:{depth}")
        depth = depth.to(device)

        # print(f"AFTER TO DEVICE depth:{depth}")
        image = image.to(device)

        image = torch.autograd.Variable(image, requires_grad=False)
        #indices_with_nans = depth.isnan().nonzero()
        #mask_out_nans = ~depth.isnan()
        #depth = torch.masked_select(depth, mask=mask_out_nans, fill = 0)
        mask_out_nans = depth.isnan()
        num_nans = (~mask_out_nans).sum()
        num_nans = torch.autograd.Variable(num_nans, requires_grad=False)
        depth[mask_out_nans] = 0.
        depth = torch.autograd.Variable(depth, requires_grad=False)
        # below the usage of 1 corresponds to dept.size(1), but they know it's 1 coz depth
        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().to(device)

        ones = torch.autograd.Variable(ones, requires_grad=False)
        optimizer.zero_grad()
        # print("GPU VRAM before model call:",torch.cuda.mem_get_info())
        output = model(image)
        output[mask_out_nans] = 0.
        #output = torch.masked_select(output, mask=mask_out_nans);

        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        # depth_normal = F.normalize(depth_normal, p=2, dim=1)
        # output_normal = F.normalize(output_normal, p=2, dim=1)
        # print(f"output:{output}")
        # print(f"depth:{depth}")

        loss_depth = torch.log(torch.abs(output - depth) + 0.5).sum()/num_nans #.mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).sum()/num_nans #.mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).sum()/num_nans#.mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).sum()/num_nans #.mean()
        """
        loss_depth = (torch.abs(output - depth) + 0.5).sum() / num_nans  # .mean()
        loss_dx = (torch.abs(output_grad_dx - depth_grad_dx) + 0.5).sum() / num_nans  # .mean()
        loss_dy = (torch.abs(output_grad_dy - depth_grad_dy) + 0.5).sum() / num_nans  # .mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).sum() / num_nans  # .mean()
        """
        # print(f"loss_depth:{loss_depth}")
        # print(f"loss_dx:{loss_dx}")
        # print(f"loss_dy:{loss_dy}")
        # print(f"loss_normal:{loss_normal}")

        loss = loss_depth + loss_normal + (loss_dx + loss_dy)

        losses.update(loss.item(), image.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        batchSize = depth.size(0)

        print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f}) , loss_depth {loss_depth}, loss_normal {loss_normal}, loss_dx {loss_dx},  loss_dy {loss_dy}'
            .format(epoch, i+1, len(train_loader), batch_time=batch_time, loss=losses, loss_depth=loss_depth, loss_normal=loss_normal, loss_dx=loss_dx, loss_dy=loss_dy
                    ))

        if loss.isnan().any():
            # exit()
            print("=====NAN VALUE IN LOSS !!!!! =====================")

def validation(batch,model):
    cos = nn.CosineSimilarity(dim=1, eps=0)
    model.eval()
    with torch.no_grad():
        if(torch.cuda.is_available()):
            get_gradient = sobel.Sobel(1).cuda()
        else:
            get_gradient = sobel.Sobel(1).cpu()

        # predict model on first sample from loader
        image, depth = batch['image'], batch['depth']
        depth = depth.to(device)
        image = image.to(device)
        # predict model on first sameple from testing
        image = torch.autograd.Variable(image, requires_grad=False)
        mask_out_nans = depth.isnan()
        num_nans = (~mask_out_nans).sum()
        num_nans = torch.autograd.Variable(num_nans, requires_grad=False)
        depth[mask_out_nans] = 0.
        depth = torch.autograd.Variable(depth, requires_grad=False)
        # below the usage of 1 corresponds to dept.size(1), but they know it's 1 coz depth
        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().to(device)

        ones = torch.autograd.Variable(ones, requires_grad=False)
        # print("GPU VRAM before model call:",torch.cuda.mem_get_info())
        output = model(image)
        mask_out_nans = output.isnan()
        output[mask_out_nans] = 0.
        #output = torch.masked_select(output, mask=mask_out_nans);

        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)


        loss_depth = torch.log(torch.abs(output - depth) + 0.5).sum()/num_nans #.mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).sum()/num_nans #.mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).sum()/num_nans#.mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).sum()/num_nans #.mean()
        
        """
        loss_depth = (torch.abs(output - depth) + 0.5).sum() / num_nans  # .mean()
        loss_dx = (torch.abs(output_grad_dx - depth_grad_dx) + 0.5).sum() / num_nans  # .mean()
        loss_dy = (torch.abs(output_grad_dy - depth_grad_dy) + 0.5).sum() / num_nans  # .mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).sum() / num_nans  # .mean()
        """

        loss = loss_depth + loss_normal + (loss_dx + loss_dy)
        
        return {"output":output,"loss_depth" :loss_depth.item(),"loss_dx":loss_dx.item(),
        "loss_dy":loss_dy.item(),"loss_normal":loss_normal.item(),"loss":loss.item() }

def save_results(results:dict,filename:str=""):
    
    # We want for training to save all epochs and outputs. 
    path = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] +"../outputs/results/"

    results_filename = path+filename+"-results.csv"

    with open(results_filename,mode="a") as file:
        w = csv.DictWriter(file, results.keys())
        w.writerow(results)



def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpointapple.pth.tar'):
    # now = datetime.datetime.now()
    # filename = f"./checkpointapple-{str(now.strftime('%m-%d-%Y-%H-%M-%S'))}.pth.tar"
    torch.save(state, filename)


if __name__ == '__main__':
    main()
