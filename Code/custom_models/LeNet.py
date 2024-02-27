import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask

class VanillaLeNet(nn.Module):
    def __init__(self, in_chan=3, num_classes=3, imsize=160+1, kernel_size=5, N=None):
        super(VanillaLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.mask = build_mask(imsize, margin=1)

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
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
        
        # check device for model:
        device = self.dummy.device
        mask = self.mask.to(device=device)
        
        x = x*mask
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x