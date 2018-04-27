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
from capsule_network import *
import argparse
import pdb
#import matplotlib.pyplot as plt
import random
import cv2

USE_CUDA=torch.cuda.is_available()





# G(z)i

class generator(nn.Module):
    # initializers
    def __init__(self, d=128,img_size=32):
        super(generator, self).__init__()
        self.img_size=img_size
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        
        if self.img_size==64:
            self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
            self.deconv4_bn = nn.BatchNorm2d(d)
            self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        if self.img_size==32: 
            self.deconv4= nn.ConvTranspose2d(d*2,1,4,2,1)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        if self.img_size==64:
            x = F.relu(self.deconv4_bn(self.deconv4(x)))
            x = F.tanh(self.deconv5(x))
        if self.img_size==32:
            x= F.tanh(self.deconv4(x))
        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128,img_size=32,dataset='mnist'):
        super(discriminator, self).__init__()
        self.output_channels=1
        if dataset=='cifar10':
            self.output_channels=3

        self.img_size=img_size
        self.conv1 = nn.Conv2d(self.output_channels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        
        if self.img_size==64:
            self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
            self.conv4_bn = nn.BatchNorm2d(d*8)
            self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        if self.img_size==32:
            self.conv4 = nn.Conv2d(d*4,1,4,1,0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        if self.img_size==64:
            x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
            x = F.sigmoid(self.conv5(x))
        if self.img_size==32:
            x=F.sigmoid(self.conv4(x))
        return x



def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise

if USE_CUDA:
    fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)
else:
    fixed_z_ = Variable(fixed_z_, volatile=True)


def save_result(path = 'result.png', isFix=False,G=None):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)

    if USE_CUDA:
        z_ = Variable(z_.cuda(), volatile=True)
    else:
        z_ = Variable(z_, volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    vutils.save_image(test_images.data,path)




# data_loader
def get_data(batch_size=64,i_size=32):
    transform = transforms.Compose([
            transforms.Scale(i_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    return train_loader

def rotate_batch(img_batch,degree_angle=0,random_bool=False,random_range=[-45,45]):
    img_batch=img_batch.numpy()
    cols, rows= img_batch.shape[2:4]
    background=np.zeros((cols*2,rows*2))
    background.fill(-1)

    for i in range(len(
        img_batch)):
        background[int(cols*0.5):int(cols*0.5)+cols,int(rows*0.5):int(rows*0.5)+rows]=img_batch[i,0,:,:]
        img=background
        if random_bool:
            degree_angle=random.uniform(random_range[0],random_range[1])

        cols2,rows2= img.shape
        M=cv2.getRotationMatrix2D((cols2/2,rows2/2),degree_angle,1)
        img = cv2.warpAffine(img,M,(cols2,rows2))
        img_batch[i,0,:,:] = img[int(cols*0.5):int(cols*0.5)+cols,int(rows*0.5):int(rows*0.5)+rows]
        
        #plt.imshow(img_batch[i,0,:,:])
      
    return torch.from_numpy(img_batch)



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


    # network
    if USE_CAPS_D:
        D=CapsNet(reconstruction_bool=reconstruction_loss_bool,
                    param=D_param,
                    SN_bool=SN_bool) #already initlialized
    else:
        D = discriminator(d=128)
        D.weight_init(mean=0.0, std=0.02)

    #generator
    G = generator(d=128)
    G.weight_init(mean=0.0, std=0.02)


    if USE_CUDA:
        G=G.cuda()
        D=D.cuda()

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # results save folder
    if not os.path.isdir('MNIST_DCGAN_results'):
        os.mkdir('MNIST_DCGAN_results')
    if not os.path.isdir('MNIST_DCGAN_results/Random_results'):
        os.mkdir('MNIST_DCGAN_results/Random_results')
    if not os.path.isdir('MNIST_DCGAN_results/Fixed_results'):
        os.mkdir('MNIST_DCGAN_results/Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_iter_ptimes'] = []
    train_hist['total_ptime'] = []
    num_iter = 0

    if verbose:
        print('training start!')


    start_time = time.time()
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        
        for x_, _ in train_loader:
            iter_start_time = time.time()
            # train discriminator D
            D.zero_grad()

            mini_batch = x_.size()[0]

            if rotate_bool:
                x_=rotate_batch(img_batch=x_,random_bool=True,random_range=[-rotate_degree_range,rotate_degree_range])


            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            if USE_CUDA:
                x_, y_real_, y_fake_ = Variable(x_).cuda(), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            else:
                x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)

            #D_result = D(x_).squeeze()
            D_result = D(x_)
            
            #D_real_loss= D.margin_loss(D_result,y_real_)
            D_real_loss=0
            
            D_real_loss= D.loss(data=x_,x=D_result[0],target=y_real_,reconstructions=D_result[1])  if USE_CAPS_D else BCE_loss(D_result, y_real_)  
   

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_= Variable(z_.cuda()) if USE_CUDA else Variable(z_)

            G_result = G(z_)
            #D_result = D(G_result).squeeze()

            D_result=D(G_result)
            
            #D_fake_loss = D.margin_loss(D_result,y_fake_)
            D_fake_loss=0

            
            D_fake_loss= D.loss(data=Variable(G_result.data,volatile=True),x=D_result[0],target=y_fake_,reconstructions=D_result[1]) if USE_CAPS_D else BCE_loss(D_result, y_fake_)

            D_fake_score = D_result[0].data.mean()
            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            # D_losses.append(D_train_loss.data[0])
            D_losses.append(D_train_loss.data[0])

            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            D.eval()
            z_= Variable(z_.cuda()) if USE_CUDA else Variable(z_)

            G_result = G(z_)
            #D_result = D(G_result).squeeze()
            D_result = D(G_result)
            
            #G_train_loss=D.margin_loss(D_result,y_real_)
            G_train_loss= D.loss(data=Variable(G_result.data,volatile=True),x=D_result[0],target=y_real_,reconstructions=D_result[1]) if USE_CAPS_D else BCE_loss(D_result, y_real_)
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data[0])
            num_iter += 1
             
            rotate_tag='_rotate_'+str(rotate_degree_range) if rotate_bool else ''
            D_tag= 'CAPS' if USE_CAPS_D else 'BASE'
            tag= hyperparam_tag +"_"+ str(num_iter) + '_size_'+str(img_size)+"_bs_"+str(batch_size)+D_tag+rotate_tag

            iter_end_time = time.time()
            per_iter_ptime = iter_end_time - iter_start_time

            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)
            train_hist['per_iter_ptimes'].append(per_iter_ptime)

            if num_iter%100==0 and SAVE_IMAGE:

                p = 'MNIST_DCGAN_results/Random_results/MNIST_'+tag+'.png'
                fixed_p = 'MNIST_DCGAN_results/Fixed_results/MNIST_'+tag+'.png'
                save_result(fixed_p,isFix=True,G=G)
                save_result(p,isFix=False,G=G)


            if verbose:
                print('Iter: [%d/%d] loss_d: %.3f loss_g: %.3f condition: %s' %  (num_iter,num_iter_limit,D_train_loss.data[0],G_train_loss.data[0],D_tag))
            
            if num_iter>=num_iter_limit and SAVE_TRAINING:
                p = 'MNIST_DCGAN_results/Random_results/MNIST_'+tag+'.png'
                fixed_p = 'MNIST_DCGAN_results/Fixed_results/MNIST_'+tag+'.png'

                save_result(fixed_p,isFix=True,G=G)
                save_result(p,isFix=False,G=G)

                #torch.save(G.state_dict(), "MNIST_DCGAN_results/generator_param_"+tag+".pkl")
                #torch.save(D.state_dict(), "MNIST_DCGAN_results/discriminator_param_param_"+tag+".pkl")
                with open('MNIST_DCGAN_results/train_hist_'+tag+'.pkl', 'wb') as f:
                    pickle.dump(train_hist, f)
                return       



    return
