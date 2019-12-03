import tensorflow as tf
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

class SuperUNet(nn.Module):
  def convolvePath(self, in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=3,  padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )

  def cropConcat(self, upsampled, contraction):
    _, _, a, b = upsampled.size()
    _, _, i, j = contraction.size()
    #u_size = min(a,b)
    y = contraction[:, :, (i-a)//2 : (i+a)//2,
                          (j-b)//2 : (j+b)//2]
    
    return torch.cat([upsampled, y], 1)


  def __init__(self, in_channel=3, out_channel=3, scale_factor=3):
    super(SuperUNet, self).__init__()
    # Contraction track
    self.contract1 = self.convolvePath(in_channel, 64)
    self.contract2 = self.convolvePath(64, 128)
    self.contract3 = self.convolvePath(128, 256)
    self.contract4 = self.convolvePath(256, 512)
    self.contract5 = self.convolvePath(512, 1024)

    self.upsample4 = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=1)
    self.upsample3 = nn.ConvTranspose2d(512, 256, kernel_size=1, stride=1)
    self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1)
    self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=scale_factor,
                                        stride=scale_factor)

    # Expansion track
    self.expand4 = self.convolvePath(1024, 512)
    self.expand3 = self.convolvePath(512, 256)
    self.expand2 = self.convolvePath(256, 128)
    self.expand1 = nn.Sequential(
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, out_channel, 1)
    )

    pass

  def forward(self, image):
    ############# Contraction #############
    c1 = self.contract1(image)
    c2 = self.contract2(c1)
    c3 = self.contract3(c2)
    c4 = self.contract4(c3)
    c5 = self.contract5(c4)
    
    ############## Expansion ###############
    u4 = self.upsample4(c5)
    e4 = self.expand4(self.cropConcat(u4, c4))

    u3 = self.upsample3(e4)
    e3 = self.expand3(self.cropConcat(u3, c3))

    u2 = self.upsample2(e3)
    e2 = self.expand2(self.cropConcat(u2, c2))

    u1 = self.upsample1(e2)
    e1 = self.expand1(u1)

    return e1
