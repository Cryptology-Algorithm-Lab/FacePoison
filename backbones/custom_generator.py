# Motivated by One for All: A Universal Generator for Concept Unlearnability via Multi-Modal Alignment

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ic, oc, s = 1):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, 3, s, 1)
        self.norm = nn.GroupNorm(oc, oc)
        self.act = nn.ReLU()
        self.shortcut = None
        
        if s > 1 or ic != oc:
            self.shortcut = nn.Conv2d(ic, oc, 1, s, 0)            
        
    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)        
        if self.shortcut != None:        
            x = self.shortcut(x)        
        return x + y
        
        
class ResNetGenerator(nn.Module):
    def __init__(self, eps=16/255):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            ResBlock(64, 64),
            nn.MaxPool2d(2),
            ResBlock(64, 128),
            nn.MaxPool2d(2),
            ResBlock(128, 256),
            nn.MaxPool2d(2), 
            ResBlock(256, 512),
            nn.MaxPool2d(2) 
        )
        
        self.bottleneck = nn.Sequential(
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ResBlock(512, 256),
            nn.Upsample(scale_factor=2),
            ResBlock(256, 128),
            nn.Upsample(scale_factor=2),
            ResBlock(128, 64),
            nn.Upsample(scale_factor=2),
            ResBlock(64, 64),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
        
        self.eps = eps
        self.fp16 = True
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.encoder(x)
            x = self.bottleneck(x)
            x = self.decoder(x)            
        x = x.float()
        alpha = 2 * F.softplus(self.alpha)
        return F.tanh(x * alpha) * self.eps
#         return x
        