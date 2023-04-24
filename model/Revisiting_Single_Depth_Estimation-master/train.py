import argparse

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
from models import modules, net, resnet, densenet, senet

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


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
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
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    # print("GPU VRAM model defined:",torch.cuda.mem_get_info())
 
    if torch.cuda.device_count() == 8:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        batch_size = 64
    elif torch.cuda.device_count() == 4:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        batch_size = 32
    else:
        model = model.cuda()
        # batch_size = 4
        batch_size = 2
    # print("GPU VRAM ifs:",torch.cuda.mem_get_info())

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # print("GPU VRAM optimizer:",torch.cuda.mem_get_info())

    train_loader = loaddata.getTrainingData(batch_size)
    # print("BatchSize:",batch_size)
    # print("GPU VRAM 0:",torch.cuda.mem_get_info())
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        start_time_loop = time.time()
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch)
        end_time_loop = time.time()
        print(f"EPOCH TRAINED FOR :{end_time_loop-start_time_loop} ")
         
    end_time = time.time()
    print(f"TRAINED FOR:{end_time-start_time} ")

    
    save_checkpoint({'state_dict': model.state_dict()})


def train(train_loader, model, optimizer, epoch):
    print("GPU VRAM 1:",torch.cuda.mem_get_info())
    print(f"===== EPOCH:{epoch} ====== ")
    criterion = nn.L1Loss()
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    # print("GPU VRAM before Sobel:",torch.cuda.mem_get_info())
    get_gradient = sobel.Sobel().cuda()
    print("GPU VRAM After Sobel:",torch.cuda.mem_get_info())

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        print("DOING ITERATION:",i)
        print("Sample batch:",sample_batched)
        # print("GPU VRAM:",torch.cuda.mem_get_info())
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda()
        image = image.cuda()
        if depth.isnan().any():
            print("DOING ITERATION:",i)
            print("Sample batch:",sample_batched)
        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)
        # print("GPU VRAM after autograd:",torch.cuda.mem_get_info())
        # below the usage of 1 corresponds to dept.size(1), but they know it's 1 coz depth
        ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
        print(f"ones.shape: {ones.shape}")
        print(f"depth.shape:{depth.shape}")
        
        ones = torch.autograd.Variable(ones)
        optimizer.zero_grad()
        # print("GPU VRAM before model call:",torch.cuda.mem_get_info())
        output = model(image)
        # print("GPU VRAM output:",torch.cuda.mem_get_info())

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
        print(f"output:{output}")
        print(f"depth:{depth}")

        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

        print(f"loss_depth:{loss_depth}")
        print(f"loss_dx:{loss_dx}")
        print(f"loss_dy:{loss_dy}")
        print(f"loss_normal:{loss_normal}")
        
        loss = loss_depth + loss_normal + (loss_dx + loss_dy)
        
        losses.update(loss.item(), image.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
   
        batchSize = depth.size(0)

        print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})'
          .format(epoch, i+1, len(train_loader), batch_time=batch_time, loss=losses))
        if depth.isnan().any():
            exit(1)
 

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
    now = datetime.datetime.now()
    filename = f"./checkpointapple-{str(now.strftime('%m-%d-%Y-%H-%M-%S'))}.pth.tar"
    torch.save(state, filename)


if __name__ == '__main__':
    main()
