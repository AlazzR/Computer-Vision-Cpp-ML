#%%
import torchvision
import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import time 
from sklearn.metrics import confusion_matrix
#%%
# VGG Model from scratcg
class VGG13(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(VGG13, self).__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(3, 64, (3, 3), 1)
        self.conv2 = torch.nn.Conv2d(64, 64, (3, 3), 1)
        self.pool1 = torch.nn.MaxPool2d((2, 2), 2)#non-overlapping
        self.conv3 = torch.nn.Conv2d(128, 128, (3, 3), 1)
        self.conv4 = torch.nn.Conv2d(128, 128, (3, 3), 1)
        self.pool2 = torch.nn.MaxPool2d((2, 2), 2)#non-overlapping
        self.conv5 = torch.nn.Conv2d(256, 256, (3, 3), 1)
        self.conv6 = torch.nn.Conv2d(256, 256, (3, 3), 1)
        self.pool3 = torch.nn.MaxPool2d((2, 2), 2)#non-overlapping
        self.conv7 = torch.nn.Conv2d(512, 512, (3, 3), 1)
        self.conv8 = torch.nn.Conv2d(512, 512, (3, 3), 1)
        self.pool4 = torch.nn.MaxPool2d((2, 2), 2)#non-overlapping
        self.conv9 = torch.nn.Conv2d(512, 512, (3, 3), 1)
        self.conv10 = torch.nn.Conv2d(512, 512, (3, 3), 1)
        self.pool5 = torch.nn.MaxPool2d((2, 2), 2)#non-overlapping

        self.fc1 = torch.nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(F.pad(x, (1, 1, 1, 1)))
        x = self.conv2(x)
        x = torch.relu(F.pad(x, (1, 1, 1, 1)))
        x = self.pool1(x)
        #print(x.shape)
        x = self.conv3(x)
        x = torch.relu(F.pad(x, (1, 1, 1, 1)))
        x = self.conv4(x)
        x = torch.relu(F.pad(x, (1, 1, 1, 1)))
        x = self.pool2(x)
        #print(x.shape)

        x = self.conv5(x)
        x = torch.relu(F.pad(x, (1, 1, 1, 1)))
        x = self.conv6(x)
        x = torch.relu(F.pad(x, (1, 1, 1, 1)))
        x = self.pool3(x)
        #print(x.shape)

        x = self.conv7(x)
        x = torch.relu(F.pad(x, (1, 1, 1, 1)))
        x = self.conv8(x)
        x = torch.relu(F.pad(x, (1, 1, 1, 1)))
        x = self.pool4(x)
        #print(x.shape)

        x = self.conv9(x)
        x = torch.relu(F.pad(x, (1, 1, 1, 1)))
        x = self.conv10(x)
        x = torch.relu(F.pad(x, (1, 1, 1, 1)))
        x = self.pool5(x)
        #print(x.shape)

        x = x.view(-1, 512 * 7 * 7)
        #print(x.shape)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        x = F.softmax(x, dim=2)

        return x
    
    @staticmethod
    def init_param(layer):
        if type(layer) == torch.nn.Conv2d:
            torch.nn.init.xavier_normal_(layer.weight)
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
    
    def initialize_parameters(self, verbose=False):
        self.apply(self.init_param)
        numberParameters = 0
        for p in self.parameters():
            numberParameters += p.numel() if p.requires_grad else 0
        if verbose:
            counter = 0
            for param in self.parameters():
                print(f"Layer {counter}")
                print(param)
                counter += 1
        print("Number of parameters is {:,}".format(numberParameters))

#%%
# Pretrained model
class VggWithCustomLayers(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(VggWithCustomLayers, self).__init__(*args, **kwargs)
        self.vgg = torchvision.models.vgg16(pretrained=True)
        for param in self.vgg.parameters():
            param.requires_grad = False
        in_features = self.vgg.classifier[-1].in_features
        block = torch.nn.Sequential(OrderedDict([
            ("conv_1", torch.nn.Linear(in_features, 2)),
            #("non-linear-1", torch.nn.ReLU()),
            #("last", torch.nn.Linear(128, 1)),
            ("softmax", torch.nn.Softmax(dim=1))
        ]))
        self.vgg.classifier[-1] = block

    def forward(self, x):
        x = self.vgg(x)
        return x