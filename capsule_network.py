import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from src.snlayers.snconv2d import *
import pdb
USE_CUDA=torch.cuda.is_available()
class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, SN_bool=False):
        super(ConvLayer, self).__init__()
        if SN_bool:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=1
                                 )
        else:

            self.conv = SNConv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       stride=1
                     )


    def forward(self, x):
        return F.relu(self.conv(x))




class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2, SN_bool=False):
        super(PrimaryCaps, self).__init__()

        self.kernel_size=kernel_size
        self.stride=stride

        if SN_bool:
            self.capsules = nn.ModuleList([
                SNConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=0) 
                              for _ in range(num_capsules)])
        else:
            self.capsules = nn.ModuleList([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=0) 
                              for _ in range(num_capsules)])
    def forward(self, x):
        input_size=x.shape[2]
        output_size=int(((input_size-self.kernel_size)/self.stride) +1)
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        batch_size=x.size(0)

        u = u.view(batch_size, 32 * output_size * output_size, -1)
        return self.squash(u)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class DigitCaps(nn.Module):
    #in_channels is the length of the capsule, while teh outchannels is the legnth of the digitcaps
    def __init__(self, num_capsules=1, num_routes=32 * 8 * 8, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1636),
            nn.ReLU(inplace=True),
            nn.Linear(1636, 1024),
            nn.Sigmoid()
        )
        
    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)
        
        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        
        reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 1, 32, 32)
        
        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self,
                reconstruction_bool=False,SN_bool=False,param=[0.9,0.1,0.5,0.005]):

        super(CapsNet, self).__init__()
        self.param=param
        self.reconstruction_bool=reconstruction_bool
        self.conv_layer = ConvLayer(SN_bool=SN_bool)
        self.primary_capsules = PrimaryCaps(SN_bool=SN_bool)
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, data):

        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))

        if self.reconstruction_bool:
            reconstructions, masked = self.decoder(output, data)
            return output, reconstructions, masked

        return output
        #return torch.sqrt((output**2).sum(dim=2, keepdim=True))

            
    def loss(self, data, x, target, reconstructions,loss_type=0):
        r_loss=0
        if loss_type==0:
            if self.reconstruction_bool:
                r_loss=self.reconstruction_loss(data, reconstructions)

            return self.margin_loss(x, target) + r_loss

        if loss_type==1:
            return torch.sqrt((x**2).sum(dim=2, keepdim=True)).mean(0).view(1)


    def margin_loss(self, x, labels, size_average=True):
        batch_size = len(x)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        left = F.relu(self.param[0] - v_c).view(batch_size, -1)
        right = F.relu(v_c - self.param[1]).view(batch_size, -1)

        loss = labels * left*left + self.param[2] * (1.0 - labels) * right*right
        loss = loss.sum(dim=1).mean()

        return loss
    
    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * self.param[3]

