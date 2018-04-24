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
    def __init__(self, d=128,img_size=32):
        super(discriminator, self).__init__()
        self.img_size=img_size
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
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
        datasets.MNIST('data', train=True, download=True, transform=transform),
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
            lambda_=10):

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
    #BCE_loss = nn.BCELoss()

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
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    num_iter = 0

    if verbose:
        print('training start!')


    start_time = time.time()
    num_gen_iter=1000
    gen_iter=0
    disc_iter=0
    num_disc_iter=5
    
    while gen_iter < num_gen_iter: #iterate throught eh gen_iter
        D_losses = []
        G_losses = []

        for x_, _ in train_loader:

         
            disc_iter+=1

            # train discriminator D
            # pdb.set_trace()
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

            




            #adding gradient penalty


            pdb.set_trace()

            alpha=torch.FloatTensor([1]).uniform_(0,1)

            difference= G_result.data - x_.data

            interpolate= Variable(x_.data + alpha*difference, requires_grad=True)

            D_interpolate= D(interpolate)

            D_interpolate_loss=D.loss(data=Variable(interpolate.data,volatile=True),x=D_interpolate[0],target=y_real_,reconstructions=D_interpolate[1])

            #D_interpolate_loss=D.loss(data=Variable(G_result.data,volatile=True),x=D_result[0],target=Variable(y_real_.data+(y_fake_-y_real_)*alpha,require_grad=True),reconstructions=D_result[1])+\

            D_interpolate_loss.backward()

            
            interpolate_grad=interpolate.grad

            grad_penalty=lambda_*((interpolate_grad**2).sum()-1)**2

            grad_penalty=Variable(grad_penalty, volatile=False)

            D_train_loss = D_real_loss + D_fake_loss + grad_penalty


            D_train_loss.backward()
            D_optimizer.step()

            # D_losses.append(D_train_loss.data[0])
            D_losses.append(D_train_loss.data[0])

            # train generator G

            if disc_iter==num_disc_iter: #start the generator training brooooooo
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
                gen_iter+=1

            #for m in range(num_m_inters):


            if gen_iter%100==0 and USE_CAPS_D and SAVE_IMAGE:
                tag=hyperparam_tag +"_"+ str(gen_iter) + '_size_'+str(img_size)+"_bs_"+str(batch_size)+'_caps'
                p = 'MNIST_DCGAN_results/Random_results/MNIST_DCGAN_'+tag+'.png'
                fixed_p = 'MNIST_DCGAN_results/Fixed_results/MNIST_DCGAN_'+tag+'.png'


                save_result(fixed_p,isFix=True,G=G)
                save_result(p,isFix=False,G=G)

           


            if verbose:
                print('gen_iters: [%d/%d] disc_iter: [%d/%d] loss_d: %.3f loss_g: %.3f grad_penalty: %.3f' %  (gen_iter,num_gen_iter,disc_iter,num_disc_iter,D_train_loss.data[0],G_train_loss.data[0],grad_penalty))
            
            if gen_iter>=num_iter_limit and SAVE_TRAINING:
                tag=hyperparam_tag +"_"+ str(gen_iter) + '_size_'+str(img_size)+"_bs_"+str(batch_size)+'_caps'
                p = 'MNIST_DCGAN_results/Random_results/MNIST_DCGAN_'+tag+'.png'
                fixed_p = 'MNIST_DCGAN_results/Fixed_results/MNIST_DCGAN_'+tag+'.png'

                save_result(fixed_p,isFix=True,G=G)
                save_result(p,isFix=False,G=G)

                torch.save(G.state_dict(), "MNIST_DCGAN_results/generator_param_"+tag+".pkl")
                torch.save(D.state_dict(), "MNIST_DCGAN_results/discriminator_param_param_"+tag+".pkl")
                with open('MNIST_DCGAN_results/train_hist_'+tag+'.pkl', 'wb') as f:
                    pickle.dump(train_hist, f)

                return

            if disc_iter==num_disc_iter:
                disc_iter=0

        gen_iter_end_time = time.time()
        per_gen_iter_time = gen_iter_end_time - gen_iter_start_time
        if verbose:
            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_gen_iter_time, torch.mean(torch.FloatTensor(D_losses)),
                                                                  torch.mean(torch.FloatTensor(G_losses))))   
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_gen_iter_time)


    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)
    

    if verbose:
        print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
        print("Training finish!")

    return



train_loader=get_data(batch_size=64,i_size=32)


run_model(lr=0.002,
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
            train_loader=train_loader, 
            hyperparam_tag='1')
