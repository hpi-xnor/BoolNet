"""
@author: nianhui guo
@license: (C) Copyright 2020-2020, NWPU
@contact: guonianhui199512@gmail.com
@software: BNN
@file: resent.py
@time: 2020/10/17 14:22
@desc:Binary Neural Network Optimization
"""

import sys
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from .modules import BConv, MultiBConv, GhostSign, GhostBNSign


__all__ = ['boolnet18', 'boolnet34']

    
def init_model(model):
    for m in model.modules():
                
        if isinstance(m, (nn.Conv2d)):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
          
          if isinstance(m, BConv):
            m.weight.data.clamp_(-0.99, 0.99)
            
          if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, (BConv, MultiBConv)):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
          
          m.weight.data.mul_(1e-2)
          #m.weight.data.clamp_(-0.5, 0.5)
          
          if m.temperature is not None:
            nn.init.constant_(m.temperature, 1)
            
        elif isinstance(m, GhostSign):
          
          if m.temperature is not None:
             nn.init.constant_(m.temperature, 1)
          
          #if m.length_1 is not None:
          #   nn.init.constant_(m.length_1, 1)
              
          #if m.length_2 is not None:
          #   nn.init.constant_(m.length_2, 1)
                            
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
              if m.weight is not None:
                nn.init.constant_(m.weight, 1)
                  
              if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
              if m.running_mean is not None:
                nn.init.constant_(m.running_mean, 0)
                
              if m.running_var is not None:
                nn.init.constant_(m.running_var, 1)    


def OR(x, y):                                  # -1,1
    y_l = y.add(1).div(2)                        # 0,1
    x_l = x.add(1).div(2)                        # 0,1

    return ((x_l.add(y_l).clamp(0,1).mul(2).add(-1)))

def XOR(x, y):                              #-1,1/1,-1
    y = XNOR(x, -y)

    return y

def XNOR(x, y):                             #-1,1

    return x.mul(y)


class Conv1x1(nn.Module):
  def __init__(self,
      in_channels = 3,
      out_channels = 64,
      stride = 1,
      dilation = 1,
      groups = 1,
      bias = False):
      super(Conv1x1, self).__init__()
        
      self.Conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride = stride, padding = 0, groups = groups, bias = bias) 
  
  def forward(self, x):
      return self.Conv1x1(x)
                                    
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels = 16, out_channels = 16, stride = 1, dilation = 1, groups = 1, bias = False, downsample=None, base_width = 64, last_block = False, max_slices = 4):
        super(BasicBlock, self).__init__()
        self.max_slices = max_slices
       
        stochastic = False

        self.conv_slices_1 = 2**(random.randint(0,int(math.log(max_slices)/math.log(2)))) if stochastic else self.max_slices  #specify an int even number to tell the slices to be used(less than max_slices)
        self.conv_slices_2 = 2**(random.randint(0,int(math.log(max_slices)/math.log(2)))) if stochastic else self.max_slices #specify an int even number to tell the slices to be used(less than max_slices)
        
        self.stride = stride
        
        self.in_channels = in_channels
        
        self.conv1 = MultiBConv(self.in_channels, out_channels, 3, stride, 1, dilation, groups = self.conv_slices_1, bias = bias, wb = True)  
        self.conv2 = MultiBConv(out_channels, out_channels, 3, 1, 1, dilation, groups = self.conv_slices_2, bias = bias, wb = True)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.ghostsign1 = GhostSign(out_channels, slices = max_slices)
        self.ghostsign2 = GhostSign(out_channels, slices = max_slices)

                
        self.downsample = downsample
        
        self.last_block = last_block
        
        
        self.stride = stride
        
    def forward(self, x):
        N, S, C, H, W = x.size()
        
        residual = x

        if self.downsample is not None:
            residual = self.downsample(residual.view(N, -1, H, W))
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ghostsign1(x)
        x = XNOR(x, residual) 
        
        
        z = self.conv2(x)
        z = self.bn2(z)
        
        if not self.last_block:
          z = self.ghostsign2(z)
          z = OR(z, residual)

        else:
          z = z + self.bn3(x.mean(1))    
        
        return z

class BoolNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,

                 groups=1, width_per_group=64, replace_stride_with_dilation=None,

                 max_slices = 4, binary_downsample=False):

        super(BoolNet, self).__init__()    
        expansion = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        scale = 1                     
        self.groups = groups
        self.base_width = width_per_group
        self.binary_downsample = binary_downsample
        
        self.inplanes = 64*scale
        self.dilation = 1
        
        self.max_slices = max_slices
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias = False)

        self.maxpool = nn.Sequential(
                            nn.BatchNorm2d(self.inplanes),
                            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
                            )
          
        self.ghostbnsign = nn.Sequential(
                            nn.BatchNorm2d(self.inplanes),
                            GhostSign(self.inplanes, slices = max_slices)
                            )

        self.layer1 = self._make_layer(block, 64*scale, layers[0], max_slices = self.max_slices)

        self.layer2 = self._make_layer(block, 128*scale, layers[1], stride=2,

                                       dilate=replace_stride_with_dilation[0], max_slices = self.max_slices)

        self.layer3 = self._make_layer(block, 256*scale, layers[2], stride=2,

                                       dilate=replace_stride_with_dilation[1], max_slices = self.max_slices)

        self.layer4 = self._make_layer(block, 512*scale, layers[3], stride=2,

                                       dilate=replace_stride_with_dilation[2], max_slices = self.max_slices)
        
        self.layer4[-1].last_block = True
        
        self.prelu2 = nn.PReLU(512* scale * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(512* scale * block.expansion, num_classes)
        
        # Zero-initialize the last BN in each residual branch,

        # so that the residual branch starts with zeros, and each residual block behaves like an identity.

        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_residual:

            for m in self.modules():

                if isinstance(m, BasicBlock):
                	nn.init.constant_(m.bn2.weight, 1e-8)
        
        init_model(self)
        
        self.distillation_loss = 0
        
        self.name = 'boolnet'
                
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, max_slices = 4):

        downsample = None

        previous_dilation = self.dilation

        if dilate:

            self.dilation *= stride

            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.binary_downsample:
                downsample = nn.Sequential(
                    BConv(self.inplanes*max_slices, planes * block.expansion, kernel_size = 1, stride = 1, padding = 0, groups = 1),
                    nn.MaxPool2d(kernel_size = 2, stride = 2),
                    nn.BatchNorm2d(planes * block.expansion),
                    GhostSign(planes * block.expansion, slices = max_slices)
                    )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes*max_slices, planes * block.expansion, kernel_size = 1, stride = 1, padding = 0, groups = max_slices),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(planes * block.expansion),
                    GhostSign(planes * block.expansion, slices=max_slices)
                )

        layers = []

        layers.append(block(self.inplanes, planes, stride, previous_dilation, self.groups, False, downsample, max_slices = max_slices))
        
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):

            layers.append(block(self.inplanes, planes, 1, self.dilation, self.groups, False,
                                None, base_width = self.base_width, max_slices = max_slices
                                ))

        return nn.Sequential(*layers)
                                
    def forward(self, x):
        out = []
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.ghostbnsign(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.prelu2(x)
        
        out = self.fc(self.avgpool(x).contiguous().view(x.size(0), -1))
        
        return out
                 
class boolnet18(BoolNet):
    """Constructs a resnet18 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    """
    def __init__(self, block = BasicBlock, layers =  [2, 2, 2, 2], num_classes = 10, max_slices = 4, binary_downsample = False):
        super(boolnet18, self).__init__(block = block, layers = layers, num_classes = num_classes, max_slices = max_slices, binary_downsample = binary_downsample)
        
        self.name = 'boolnet18'
        
class boolnet34(BoolNet):
    """Constructs a resnet34 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    
    """
    def __init__(self, block = BasicBlock, layers =  [3, 4, 6, 3], num_classes = 10, max_slices = 4, binary_downsample = False):
        super(boolnet34, self).__init__(block = block, layers = layers, num_classes = num_classes, max_slices = max_slices, binary_downsample = binary_downsample)
        
        self.name = 'boolnet34'
