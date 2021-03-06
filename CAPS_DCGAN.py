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
import pdb, traceback, sys

USE_CUDA=torch.cuda.is_available()





# G(z)i

class generator(nn.Module):
    # initializers
    def __init__(self, d=128,img_size=32,output_channels=1):
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
            self.deconv5 = nn.ConvTranspose2d(d, output_channels, 4, 2, 1)
        if self.img_size==32: 
            self.deconv4= nn.ConvTranspose2d(d*2,output_channels,4,2,1)
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
def get_data(batch_size=64,i_size=32, mode= 'mnist'):
    transform = transforms.Compose([
            transforms.Scale(i_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    if mode=='mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)

    if mode=='cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    return train_loader


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
            dataset=None):


    if dataset != None:
        train_loader=get_data(batch_size=batch_size,i_size=img_size, mode= dataset)

    # network
    if USE_CAPS_D:
        D=CapsNet(reconstruction_bool=reconstruction_loss_bool,param=D_param,SN_bool=SN_bool,dataset=dataset,input_img_size=img_size) #already initlialized
    else:
        D = discriminator(d=128,dataset=dataset)
        D.weight_init(mean=0.0, std=0.02)

    #generator
    if dataset=='mnist':
        G = generator(d=128,output_channels=1,img_size=img_size)

    if dataset=='cifar10':
        G = generator(d=128,output_channels=3,img_size=img_size)


    G.weight_init(mean=0.0, std=0.02)


    if USE_CUDA:
        G=G.cuda()
        D=D.cuda()

    # Binary Cross Entropy loss
    #BCE_loss = nn.BCELoss()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # results save folder
    if not os.path.isdir(dataset+'_DCGAN_results'):
        os.mkdir(dataset+'_DCGAN_results')
    if not os.path.isdir(dataset+'_DCGAN_results/Random_results'):
        os.mkdir(dataset+'_DCGAN_results/Random_results')
    if not os.path.isdir(dataset+'_DCGAN_results/Fixed_results'):
        os.mkdir(dataset+'_DCGAN_results/Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    num_iter = 0

    if verbose:
        print('training start!')


    start_time = time.time()
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        for x_, _ in train_loader:
            # train discriminator D
            D.zero_grad()

            mini_batch = x_.size()[0]
           
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            if USE_CUDA:
                x_, y_real_, y_fake_ = Variable(x_).cuda(), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            else:
                x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)

            #D_result = D(x_).squeeze()
            D_result = D(x_)
            #D_real_loss = BCE_loss(D_result, y_real_)
            #D_real_loss= D.margin_loss(D_result,y_real_)

            D_real_loss= D.loss(data=x_,x=D_result[0],target=y_real_,reconstructions=D_result[1])        

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)

            if USE_CUDA:
                z_ = Variable(z_.cuda())
            else:
                z_ = Variable(z_)

            G_result = G(z_)

            #D_result = D(G_result).squeeze()
            D_result=D(G_result)
            #D_fake_loss = BCE_loss(D_result, y_fake_)
            #D_fake_loss = D.margin_loss(D_result,y_fake_)
            D_fake_loss= D.loss(data=Variable(G_result.data,volatile=True),x=D_result[0],target=y_fake_,reconstructions=D_result[1])

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
            if USE_CUDA:
                z_ = Variable(z_.cuda())
            else:
                z_ = Variable(z_)

            G_result = G(z_)
            #D_result = D(G_result).squeeze()
            D_result = D(G_result)
            #G_train_loss = BCE_loss(D_result, y_real_)
            #G_train_loss=D.margin_loss(D_result,y_real_)
            G_train_loss= D.loss(data=Variable(G_result.data,volatile=True),x=D_result[0],target=y_real_,reconstructions=D_result[1])
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data[0])
            num_iter += 1

            if num_iter%10==0:
                train_hist['D_losses'].append((num_iter,D_losses[-1]))
                train_hist['G_losses'].append((num_iter,G_losses[-1]))

            if num_iter%100==0 and USE_CAPS_D and SAVE_IMAGE:
                tag=hyperparam_tag +"_"+ str(num_iter) + '_size_'+str(img_size)+"_bs_"+str(batch_size)+'_caps'
                p = dataset+'_DCGAN_results/Random_results/'+dataset+'_DCGAN_'+tag+'.png'
                fixed_p = dataset+'_DCGAN_results/Fixed_results/'+dataset+'_DCGAN_'+tag+'.png'

                save_result(fixed_p,isFix=True,G=G)
                save_result(p,isFix=False,G=G)

            if verbose:
                print('epoch: [%d/%d] batch: [%d] loss_d: %.3f loss_g: %.3f' %  (epoch+1,train_epoch,num_iter,D_train_loss.data[0],G_train_loss.data[0]))
            
            if num_iter>=num_iter_limit and SAVE_TRAINING:
                tag=hyperparam_tag +"_"+ str(num_iter) + '_size_'+str(img_size)+"_bs_"+str(batch_size)+'_caps'
                p = dataset+'_DCGAN_results/Random_results/'+dataset+'_DCGAN_'+tag+'.png'
                fixed_p = dataset+'_DCGAN_results/Fixed_results/'+dataset+'_DCGAN_'+tag+'.png'

                save_result(fixed_p,isFix=True,G=G)
                save_result(p,isFix=False,G=G)

                #torch.save(G.state_dict(), "MNIST_DCGAN_results/generator_param_"+tag+".pkl")
                #torch.save(D.state_dict(), "MNIST_DCGAN_results/discriminator_param_param_"+tag+".pkl")
                with open(dataset+'_DCGAN_results/train_hist_'+tag+'.pkl', 'wb') as f:
                    pickle.dump(train_hist, f)

                return

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        #if verbose:
        #print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),torch.mean(torch.FloatTensor(G_losses))))   
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)
    
    if verbose:
        #print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
        print("Training finish!")

    return


parser = argparse.ArgumentParser()
parser.add_argument('--i', required=False, type=int, default=64, help='size of image')
parser.add_argument('--d',required=False, type=str, default='cifar10',help='dataset name')
parser.add_argument('--b',required=False, type=int, default=64,help='batch size')
opt = parser.parse_args()


try:

    run_model(lr=0.002,
                batch_size=opt.b,
                train_epoch= 20,
                img_size=opt.i, 
                SN_bool=True, 
                D_param=[0.9,0.1,0.5,0.005],
                reconstruction_loss_bool=True, 
                USE_CAPS_D=True, 
                SAVE_TRAINING=True, 
                SAVE_IMAGE=True, 
                num_iter_limit=2000, 
                verbose=True, 
                train_loader=None, 
                hyperparam_tag='1',
                dataset=opt.d)
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

