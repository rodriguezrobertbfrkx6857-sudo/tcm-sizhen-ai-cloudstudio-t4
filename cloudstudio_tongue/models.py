"""State-dict compatible frozen TongueAI architectures, without asset downloads."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from transformers import Dinov2Config,Dinov2Model

class GlobalClassifier(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone;self.classifier=nn.Linear(384,2)
    def forward(self,pixel_values):
        encoded=self.backbone(pixel_values=pixel_values,interpolate_pos_encoding=True)
        return self.classifier(encoded.last_hidden_state[:,0,:])

class DoubleConv(nn.Module):
    def __init__(self,i,o):
        super().__init__();self.layers=nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True),nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
    def forward(self,x):return self.layers(x)

class Up(nn.Module):
    def __init__(self,i,s,o):super().__init__();self.layers=DoubleConv(i+s,o)
    def forward(self,x,skip):
        x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        return self.layers(torch.cat((x,skip),dim=1))

class ResNet18UNet(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=models.resnet18(weights=None)
        self.up3=Up(512,256,256);self.up2=Up(256,128,128);self.up1=Up(128,64,64);self.up0=Up(64,64,64)
        self.out=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(32,1,1))
    def forward(self,x):
        e=self.encoder;a=e.relu(e.bn1(e.conv1(x)));b=e.layer1(e.maxpool(a));c=e.layer2(b);d=e.layer3(c);z=e.layer4(d)
        z=self.up3(z,d);z=self.up2(z,c);z=self.up1(z,b);z=self.up0(z,a)
        return self.out(F.interpolate(z,size=(z.shape[-2]*2,z.shape[-1]*2),mode='bilinear',align_corners=False))
