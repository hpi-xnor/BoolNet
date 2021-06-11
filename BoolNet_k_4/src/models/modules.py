"""
@author: nianhui guo
@license: (C) Copyright 2020-2020, NWPU
@contact: guonianhui199512@gmail.com
@software: BNN
@file: modules.py
@time: 2020/10/17 12:03
@desc:Binary Neural Network Optimization
"""
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def binarize(x, temperature = 1.0, progressive = False, scale = False):
    replace = x.clamp(-1, 1) #.clamp(-1, 1)
    
    if scale:
      mean = abs(x).mean(-1, keepdim = True).mean(-2, keepdim = True).mean(-3, keepdim = True)
    else:
      mean = 1.0
      
    with torch.no_grad():
      binary = F.hardtanh(x/max(temperature, 0.0))      
      if not progressive:
        binary = binary.sign()
      
    return ((binary - replace).detach() + replace).mul(mean)


class BConv(nn.Module):
  def __init__(self,
      in_channels = 3,
      out_channels = 64,
      kernel_size = 2,
      stride = 1,
      padding = 1,
      dilation = 1,
      groups = 1,
      bias = True,
      wb = True,
      ab = True):
    super(BConv, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.wb = wb
    self.ab = ab
    
    self.register_buffer("temperature", torch.Tensor([1]))
    
    if bias:
      self.bias = nn.Parameter(torch.zeros(out_channels))
    else:
      self.bias = None
        
    self.weight = nn.Parameter(torch.randn(out_channels, int(in_channels//groups), kernel_size, kernel_size))

    if self.padding > 0:
        self.replicatepad = nn.ReplicationPad2d(padding = (padding, padding, padding, padding))
    else:
        self.replicatepad = lambda x:x

  def update_temperature(self):
    self.temperature *= 0.965
    
  def forward(self, x):
    
    if self.wb:
      weight = binarize(self.weight, self.temperature, progressive = self.training, scale = True)
    
    x = x.view(x.size(0), -1, x.size(-2), x.size(-1))

    x = self.replicatepad(x)

    out = F.conv2d(input = x,
                  weight = weight, 
                  bias = self.bias, 
                  stride= self.stride, 
                  padding= 0,
                  dilation= self.dilation, 
                  groups = self.groups)
    
    return out
              
class MultiBConv(nn.Module):
  def __init__(self,
      in_channels = 3,
      out_channels = 64,
      kernel_size = 2,
      stride = 1,
      padding = 1,
      dilation = 1,
      groups = 1,
      bias = False,
      wb = True,
      ):
    super(MultiBConv, self).__init__()
      
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.wb = wb
    
    self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
    
    if bias:
      self.bias = nn.Parameter(torch.zeros(out_channels))
    else:
      self.bias = None
    
    if self.padding > 0:
        self.replicatepad = nn.ReplicationPad2d(padding = (padding, padding, padding, padding))
    else:
        self.replicatepad = lambda x: x

    self.register_buffer("temperature", torch.Tensor([1]))
    
  def update_temperature(self):
    self.temperature *= 0.965
                   
  def forward(self, x):
    
    assert len(x.size()) == 5, "Only support multi slice input"
    
    N, S, C, H, W = x.size()

    weight = binarize(self.weight, self.temperature.to(device = "cuda" if x.is_cuda else "cpu"), progressive = self.training, scale = True)
    
    if self.groups>1:
      if S > self.groups:
        x = x[:,int((S- self.groups)//2):int((S+self.groups)//2)] 
      elif S == self.groups:
        pass
      else:
        raise ValueError("The number of slices must be larger than groups ")
      
    elif self.groups == 1:
      x = x[:,S//2].unsqueeze(1)
    
    else:
    	raise ValueError("The number of groups must be larger than one ")
    
    x = x.view(N, -1, H, W)

    x = self.replicatepad(x)

    out = F.conv2d(input = x,
                  weight = weight, 
                  bias = self.bias, 
                  stride= self.stride, 
                  padding= 0,
                  dilation= self.dilation, 
                  groups = self.groups)
    
    return out

class GhostBNSign(nn.Module):
  def __init__(self, channels, affine = True, track_running_stats = True, slices = 4, binary = True, global_binary = False, norm = "Batch"):
    super(GhostBNSign, self).__init__()
    self.channels = channels
    self.affine = affine
    self.track_running_stats = track_running_stats
    self.slices = slices
    self.binary = binary
  
    self.bn1 = nn.BatchNorm2d(channels)
    
    if not global_binary:    
      self.bn2 = nn.BatchNorm2d(channels)    
      
      if binary:
        self.adaptive_zero_points = nn.Sequential(
              BConv(channels, channels, kernel_size = 3, stride = 1, padding = 1, groups = channels),
              nn.BatchNorm2d(channels),
              nn.AdaptiveAvgPool2d((1,1)),
              )
                  
      else:
        self.adaptive_zero_points = nn.Sequential(
              nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1, bias = False, groups = channels),
              nn.BatchNorm2d(channels),
              )
    else:
      self.adaptive_zero_points = None
      
    self.GhostSign = GhostSign(channels, slices = slices)  
    
    self.register_buffer("temperature", torch.Tensor([1]))
  
  def update_temperature(self):
    self.temperature *= 0.965
    
  def forward(self, x):
    N, C, H, W = x.size()
    
    z = self.bn1(x)
    
    if self.adaptive_zero_points is not None:
      y = self.bn2(x)    
      if self.binary:
        y = binarize(y, temperature = self.temperature, progressive = False, scale = False)

      z = z.add(self.adaptive_zero_points(y))
      
    z = self.GhostSign(z)
    
    return z
                     
class GhostSign(nn.Module):
  def __init__(self, channels, slices = 4, mode = "uniform", stochastic = False):
    super(GhostSign, self).__init__()
    self.channels = channels
    assert slices == 1 or slices > 0 and (slices % 2 == 0), "the number of slics must be even"

    self.k = slices//2
    
    self.mode = mode
   
    self.register_buffer("temperature", torch.Tensor([1]))

    slice_1 = []
    slice_2 = []
    
    for i in range(-self.k, self.k+1):
      if i != 0 and self.k != 0:
        if i <0:
          index = i + self.k
          if self.mode == "uniform":
            slice_1.append(1.0/float((self.k+1))*i)
            
          elif self.mode == "non_uniform":
            slice_negtive = -(2**(-(2**int(math.log(self.k)/math.log(2))) + abs(i) - 1))
            slice_1.append(slice_negtive)
          
          else:
            raise ValueError
          
        elif i>0:
          index = i + self.k - 1
          if self.mode == "uniform":
            slice_2.append(1.0/float((self.k+1))*i)
          
          elif self.mode == "non_uniform":
            slice_positive = (2**(-(2**(int(math.log(self.k)/math.log(2)))) + abs(i) - 1))
            slice_2.append(slice_positive)  
          
          else:
            raise ValueError
        
    if len(slice_1+slice_2) != 0:
      self.slice_1  =  (torch.Tensor(slice_1).reshape(1, -1, 1, 1, 1).mul(1))
      self.slice_2 = (torch.Tensor(slice_2).reshape(1, -1, 1, 1, 1).mul(1))
    else:
      self.slice_1 = torch.zeros(1, 1, 1, 1, 1)              
      self.slice_2 = torch.zeros(1, 1, 1, 1, 1) 
    
      
  def update_temperature(self):
    self.temperature *= 0.965
 
  def forward(self, x):    
    
    if not self.k == 0:  
      slice = torch.cat(((self.slice_1.to(device = "cuda" if x.is_cuda else "cpu")), (self.slice_2.to(device = "cuda" if x.is_cuda else "cpu"))), dim = 1)
      
    else:
      slice = self.slice_1.to(device = "cuda" if x.is_cuda else "cpu")
      
    x = x.unsqueeze(1) + slice
    
    x = binarize(x, self.temperature,  progressive = False, scale = False)
    
    return x 
