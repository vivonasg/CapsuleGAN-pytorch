import os, time
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from pytorch_MNIST_CAPS_DCGAN import *



import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--n', required=False, type=int, default=1000, help='number of batches')
parser.add_argument('--b',required=False, type=int, default=32,help='batch size')
opt = parser.parse_args()


param_hyper=[[0.9],[0.1],[0.5],[0.005]]
lr_hyper=[0.0002,0.002,0.01]
SN_hyper=[False]
CAPS_bool=[True]

train_loader=get_data(batch_size=opt.b)

for param_0 in param_hyper[0]:
	for param_1 in param_hyper[1]:
		for param_2 in param_hyper[2]:
			for param_3 in param_hyper[3]:
				for lr in lr_hyper:
					for SN in SN_hyper:
						for CAPS in CAPS_bool:

							hyper_tag=str(param_0)+'-'+str(param_1)+'-'+str(param_2)+'-'+str(param_3)+'-'+str(lr)+'-'+str(SN)
							
							print(hyper_tag)

							run_model(lr=lr,
					            SN_bool=SN, 
					            D_param=[param_0,param_1,param_2,param_3],
					            num_iter_limit=opt.n,
					            hyperparam_tag=hyper_tag,
					            train_loader=train_loader,
					            USE_CAPS_D=CAPS)
