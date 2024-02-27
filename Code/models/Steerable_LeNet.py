import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn

imgsize = 160 #pixels along one side
num_classes = 3
num_channels =3

class CNSteerableLeNet(nn.Module):
    def __init__(self, in_chan=num_channels, out_chan=num_channels, imsize=imgsize+1, kernel_size=5, N=8):
        super(CNSteerableLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.r2_act = gspaces.rot2dOnR2(N)
        
        in_type = enn.FieldType(self.r2_act, in_chan*[self.r2_act.trivial_repr]) #times in_chan added
        self.input_type = in_type
        
        out_type = enn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])
        self.mask = enn.MaskModule(in_type, imsize, margin=1)
        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu1 = enn.ReLU(out_type, inplace=True)
        self.pool1 = enn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        in_type = self.pool1.out_type
        out_type = enn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.conv2 = enn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu2 = enn.ReLU(out_type, inplace=True)
        self.pool2 = enn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        
        self.gpool = enn.GroupPooling(out_type)

        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        
        self.drop  = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
      
      
    def forward(self, x):
        
        x = enn.GeometricTensor(x, self.input_type)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.gpool(x)
        x = x.tensor
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x