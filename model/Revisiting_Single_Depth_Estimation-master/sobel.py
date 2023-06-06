"""Sobel filter for edge detection"""
import torch
import torch.nn as nn
import numpy as np

class Sobel(nn.Module):
    def __init__(self, num_input_channels):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(num_input_channels, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack([np.stack([edge_kx for i in range(num_input_channels)]), np.stack([edge_ky for i in range(num_input_channels)])])

        edge_k = torch.from_numpy(edge_k).float().view(2, num_input_channels, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if(len(x.shape) != 4): #no batch
            x = x.unsqueeze(0)
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
        return out

