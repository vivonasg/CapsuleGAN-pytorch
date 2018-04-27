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
parser.add_argument('--n', required=False, type=int, default=2000, help='number of batches')
parser.add_argument('--b',required=False, type=int, default=32,help='batch size')
opt = parser.parse_args()


param_hyper=[[0.9],[0.1],[0.5],[0.005]]
lr_hyper=[0.002]
SN_hyper=[False]
CAPS_bool=[False,True]
rotate_bool=[True]
rotate_angle=[15,30,45]
'''
def run_model(lr=0.002,
            batch_size=64,
            train_epoch= 20,
            img_size=32, 
            SN_bool=True, 
            D_param=[0.9,0.1,0.5,0.005],
            reconstruction_loss_bool=True, 
            USE_CAPS_D=True, 
            SAVE_TRAINING=True, 
            SAVE_IMAGE=True, 
            num_iter_limit=5, 
            verbose=True, 
            train_loader=None, 
            hyperparam_tag='1', 
            rotate_bool=True, 
            rotate_degree_range=45):
'''


train_loader=get_data(batch_size=opt.b)

for param_0 in param_hyper[0]:
	for param_1 in param_hyper[1]:
		for param_2 in param_hyper[2]:
			for param_3 in param_hyper[3]:
				for lr in lr_hyper:
					for SN in SN_hyper:
						for CAPS in CAPS_bool:
							for rotate in rotate_bool:
								hyper_tag=str(param_0)+'-'+str(param_1)+'-'+str(param_2)+'-'+str(param_3)+'-'+str(lr)+'-'+str(SN)
								if rotate:
									for angle in rotate_angle:			
										print(hyper_tag)
										print('angle: ', angle)
										run_model(lr=lr,
								            SN_bool=SN, 
								            D_param=[param_0,param_1,param_2,param_3],
								            num_iter_limit=opt.n,
								            hyperparam_tag=hyper_tag,
								            train_loader=train_loader,
								            USE_CAPS_D=CAPS, 
								            rotate_bool=rotate,
								            rotate_degree_range=angle
								            )
								else: 		
									print(hyper_tag)
									run_model(lr=lr,
							            SN_bool=SN, 
							            D_param=[param_0,param_1,param_2,param_3],
							            num_iter_limit=opt.n,
							            hyperparam_tag=hyper_tag,
							            train_loader=train_loader,
							            USE_CAPS_D=CAPS, 
							            rotate_bool=rotate,
							            rotate_degree_range=0
							            )



