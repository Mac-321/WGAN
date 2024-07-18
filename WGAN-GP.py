# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:01:28 2023

@author: mccan
"""
import torch
import torchvision
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets,transforms
import torchvision.models as models

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
       # self.flatten = nn.Flatten()
        # 128-256-512 then use softmax 
        self.softMax = nn.Softmax(dim =1)
        self.mlp_stack = nn.Sequential(
            nn.Linear(100,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU())
        
        self.adjacencyMatrix = nn.Sequential(
             nn.Linear(512,28*28*1),
             nn.Tanh())
           
    def forward(self,x):
        x = self.mlp_stack(x)
        logits1 = self.adjacencyMatrix(x)
        logits1 =logits1.view(64,1,28,28)
        #print(logits1)
        
        return logits1

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear2 = nn.Linear(512,1)
        self.linear10 = nn.Linear(2,512)
        self.Conv1 = nn.Conv2d(1,128, 2)
        self.Conv2 = nn.Conv2d(128,256, 2)
        self.Conv3 = nn.Conv2d(256,512,2)
        self.Conv4 = nn.Conv2d(256,512,2)
        self.flatten = nn.Flatten()
        self.Tanh = nn.Tanh()
        self.ReLu = nn.ReLU()
        self.Sig = nn.Sigmoid()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28*1,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1))
        
    def forward(self,x):

          logits = self.net(x)
          #logits = self.convNet(x)
          return logits

#Uses WGAN-GP
def trainGANNetwork(dataloader,generator,discriminator,optimG,optimD):
     #set model to train
    generator.train()
    discriminator.train()
    k =1
    nCritic =5
    lamda =10     
    #j=0
    # For every batch of data in dataloader i get the data and labels associated
    # with everything in the batch 
    for i, (realDataSet,label) in enumerate(dataloader):
        #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
        #Learning Algorithm WGAN-GP from 
        #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
        #realDataSet is the X real samples
        # Z noise data samples
        # Two batches as in original paper one for Dis and Gen
        for i in range(nCritic):
            #print("Real sample",realDataSet.size())
            optimD.zero_grad()
            # Sample Z (noise)
            noise = torch.randn(64,100)
            tildaX =generator(noise)
            #print("Xtild",tildaX.size())
            # realDataSet is sample of X
            # Random uniform variable epsilon
            # Has to be done like this so epsilomn for each thing is different 
            epsilon = torch.distributions.uniform.Uniform(0,1).sample([64,1,1,1])
            #print("E",epsilon.size())
            hatX = epsilon*realDataSet
            #print("Hat x1",hatX.size())
            hatX = hatX + (1-epsilon)*tildaX
            #print(hatX.size())
            # Original WGAN calculation ####
            realData =discriminator(realDataSet)
            #print("Real data",realData.size())
            fakeData =discriminator(tildaX)
            #print("Fake data",fakeData.size())
            ##############################
            gradhatX =discriminator(hatX)
            # https://discuss.pytorch.org/t/error-grad-can-be-implicitly-created-only-for-scalar-outputs/38102/15
            gradhatX.backward(torch.ones_like(gradhatX),retain_graph= True)
            # Normalise vector in 2 D as wasserstein not 3d
            gradhatX.norm(2)
            gradPen = (gradhatX-1)*(gradhatX-1)
            L = torch.mean(fakeData) - torch.mean(realData) + lamda*torch.mean(gradPen)
            L.backward(retain_graph= True)
            optimD.step()
            #print(L)
        optimG.zero_grad()
        noise = torch.randn(64,100)
        generateData= generator(noise)
        # Minimising negative same as maxing - mentions in GAN paper
        gen = torch.mean(-discriminator(generateData))
        #print(gen)
        gen.backward(retain_graph= True)
        optimG.step()
        
        return gen , L

# Params /\/\/\/\/\/\/\/\
learningRate = 5e-5
epochs = 200
model = CNN()
optimiser = torch.optim.Adam(model.parameters(), lr = learningRate,betas= (0.0,0.9))
#/\/\/\/\/\/\/\/\/\/\/\
# data from MNIST data set converted to tensors (3D matrices)
# ToTensor() - converts PIL image into FloatTensor - scales image intensity between 0 to 1 (binary)
MNIST_data_train = datasets.MNIST(root ="data",train =True,download =True, transform =ToTensor())
MNIST_data_test = datasets.MNIST(root ="data",train =False,download =True, transform =ToTensor())

#// Putting in DataLoader allows for the iteration through a dataset
# use small batch sizes , takes a sample size of 64 then reshuffles entire
#data set -to avoid model overfitting- makes training more random?
# Use droplast to avoid any batches that are not of size 64
train_data = DataLoader(MNIST_data_train,batch_size=64,shuffle = True,drop_last = True)
test_data = DataLoader(MNIST_data_test,batch_size=64,shuffle = True,drop_last = True)
#/\/\/\/\/\/\/\/\/\/\/\
#print("Batch of 64 containing 1by 28 by images",train_feat.size())
# for i in range(epochs):
#     print("Epoch number:",i)
#     trainNetwork(train_data,model,lossFunction,optimiser)
#     testNetwork(test_data,model,lossFunction)

generator = Generator()
discriminator =Discriminator()
gen_optimiser = torch.optim.Adam(generator.parameters(), lr = learningRate,betas= (0.0,0.9))
dis_optimiser = torch.optim.Adam(discriminator.parameters(), lr = learningRate,betas= (0.0,0.9))
# # # # # #   
checkpoint = torch.load('model.pth')
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
gen_optimiser.load_state_dict(checkpoint['gen_state_dict'])
dis_optimiser.load_state_dict(checkpoint['dis_state_dict'])

generator.train()
discriminator.train()

# for i in range(epochs):
#     print("Epoch number:",i)
#     lossG, lossD =trainGANNetwork(train_data,generator,discriminator,gen_optimiser,dis_optimiser )
#     print(lossG)
#     print(lossD)
#model1 -trained for the 400
#model trained for 200 dont overwrite just yet
torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'gen_state_dict': gen_optimiser.state_dict(),
            'dis_state_dict': dis_optimiser.state_dict()
            },'model10.pth')
noise = torch.randn(64,100)
generator.eval()
discriminator.eval()
for i in range(1):
    genFakeData =generator(noise)
    #realDisTest =discriminator(train_feat[0])
    plt.imshow(genFakeData[0].detach().numpy().reshape(28,28))
    #
    plt.imshow(genFakeData[55].detach().numpy().reshape(28,28), cmap="gray")
    # D(G(z)) 
    fakeData =discriminator(genFakeData)
    print("Output",fakeData)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)