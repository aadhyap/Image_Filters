#!/usr/bin/env python3

"""

"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import tensor
from Misc.DataUtils import *
import matplotlib.pyplot as plt
import numpy as np








# Don't generate pyc codes
#sys.dont_write_bytecode = True


## functions to show an image
def main():
    #Followed Tutorial on how to do basic neural network: https://colab.research.google.com/github/omarsar/pytorch_notebooks/blob/master/pytorch_quick_start.ipynb#scrollTo=WCduzkfCCj9a
    BATCH_SIZE = 16

    ## transformations
    transform = transforms.Compose(
        [transforms.ToTensor()])

    ## download and load training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2) #batch size 32

    ## download and load testing dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)


    def imshow(img):
        #img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    ## get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    ## show images
    imshow(torchvision.utils.make_grid(images))




    for images, labels in trainloader:
        print("Image batch dimensions:", images.shape)
        print("Image label dimensions:", labels.shape)
        break

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()

            # 32x32x3 => 30x30x96
            kernel = 3
            imgsize = 32
            size = imgsize - kernel + 1
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3)
            self.d1 = nn.Linear(size * size * 96, 128)
            self.d2 = nn.Linear(128, 10)

        def forward(self, x):
            # 32x3x32x32 => 32x96x30x30
            x = self.conv1(x)
            x = F.relu(x)

            # flatten => 32 x (96*30*30)
            x = x.flatten(start_dim = 1)

            # 96 x (96*30*30) => 96x128
            x = self.d1(x)
            x = F.relu(x)
            logits = self.d2(x)
            out = F.softmax(logits, dim=1)
            return out

    ## test the model with 1 batch
    model = MyModel()
    for images, labels in trainloader:
        print("batch size:", images.shape)
        out = model(images)
        print(out.shape)
        break

    learning_rate = 0.001
    num_epochs = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyModel()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) 


    ## compute accuracy
    def get_accuracy(logit, target, batch_size):
        ''' Obtain accuracy for training round '''
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects/batch_size
        return accuracy.item()

    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_acc = 0.0

        model = model.train()

        ## training step
        for i, (images, labels) in enumerate(trainloader):
            
            images = images.to(device)
            labels = labels.to(device)

            ## forward + backprop + loss
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()

            ## update model params
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(logits, labels, BATCH_SIZE)
        
        model.eval()
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
              %(epoch, train_running_loss / i, train_acc/i))  



    #Testing
    test_acc = 0.0
    for i, (images, labels) in enumerate(testloader, 0):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        print(outputs)
        test_acc += get_accuracy(outputs, labels, BATCH_SIZE)
            
    print('Test Accuracy: %.2f'%( test_acc/i))
    
if __name__ == '__main__':
    main()
 
