import torch.nn as nn
import torch
import os
from function import adaptive_instance_normalization as adain
from function import calc_mean_std
from torch.utils.serialization import load_lua
import numpy as np

def load_param_from_t7(model, in_layer_index, out_layer):
  out_layer.weight = torch.nn.Parameter(model.get(in_layer_index).weight.float())
  out_layer.bias = torch.nn.Parameter(model.get(in_layer_index).bias.float())
load_param = load_param_from_t7

class Encoder5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder5, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
                                    [[[[0]],[[0]],[[255]]],
                                     [[[0]],[[255]],[[0]]],
                                     [[[255]],[[0]],[[0]]]])).float())
    self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
                                    [-103.939,-116.779,-123.68])).float())
    self.conv11 = nn.Conv2d(  3, 64,3,1,0) # conv1_1
    self.conv12 = nn.Conv2d( 64, 64,3,1,0) # conv1_2
    self.conv21 = nn.Conv2d( 64,128,3,1,0) # conv2_1
    self.conv22 = nn.Conv2d(128,128,3,1,0) # conv2_2
    self.conv31 = nn.Conv2d(128,256,3,1,0) # conv3_1
    self.conv32 = nn.Conv2d(256,256,3,1,0) # conv3_2
    self.conv33 = nn.Conv2d(256,256,3,1,0) # conv3_3
    self.conv34 = nn.Conv2d(256,256,3,1,0) # conv3_4
    self.conv41 = nn.Conv2d(256,512,3,1,0) # conv4_1
    self.conv42 = nn.Conv2d(512,512,3,1,0) # conv4_2
    self.conv43 = nn.Conv2d(512,512,3,1,0) # conv4_3
    self.conv44 = nn.Conv2d(512,512,3,1,0) # conv4_4
    self.conv51 = nn.Conv2d(512,512,3,1,0) # conv5_1
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.conv0)
        load_param(t7_model, 2,  self.conv11)
        load_param(t7_model, 5,  self.conv12)
        load_param(t7_model, 9,  self.conv21)
        load_param(t7_model, 12, self.conv22)
        load_param(t7_model, 16, self.conv31)
        load_param(t7_model, 19, self.conv32)
        load_param(t7_model, 22, self.conv33)
        load_param(t7_model, 25, self.conv34)
        load_param(t7_model, 29, self.conv41)
        load_param(t7_model, 32, self.conv42)
        load_param(t7_model, 35, self.conv43)
        load_param(t7_model, 38, self.conv44)
        load_param(t7_model, 42, self.conv51)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    y = self.conv0(input)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    y = self.relu(self.conv22(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv31(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv34(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv41(self.pad(y)))
    y = self.relu(self.conv42(self.pad(y)))
    y = self.relu(self.conv43(self.pad(y)))
    y = self.relu(self.conv44(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv51(self.pad(y)))
    return y

class Decoder5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder5, self).__init__()
    self.fixed = fixed
    
    self.conv51 = nn.Conv2d(512,512,3,1,0)
    self.conv44 = nn.Conv2d(512,512,3,1,0)
    self.conv43 = nn.Conv2d(512,512,3,1,0)
    self.conv42 = nn.Conv2d(512,512,3,1,0)
    self.conv41 = nn.Conv2d(512,256,3,1,0)
    self.conv34 = nn.Conv2d(256,256,3,1,0)
    self.conv33 = nn.Conv2d(256,256,3,1,0)
    self.conv32 = nn.Conv2d(256,256,3,1,0)
    self.conv31 = nn.Conv2d(256,128,3,1,0)
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 1,  self.conv51)
        load_param(t7_model, 5,  self.conv44)
        load_param(t7_model, 8,  self.conv43)
        load_param(t7_model, 11, self.conv42)
        load_param(t7_model, 14, self.conv41)
        load_param(t7_model, 18, self.conv34)
        load_param(t7_model, 21, self.conv33)
        load_param(t7_model, 24, self.conv32)
        load_param(t7_model, 27, self.conv31)
        load_param(t7_model, 31, self.conv22)
        load_param(t7_model, 34, self.conv21)
        load_param(t7_model, 38, self.conv12)
        load_param(t7_model, 41, self.conv11)
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    y = self.relu(self.conv51(self.pad(input)))
    y = self.unpool(y)
    y = self.relu(self.conv44(self.pad(y)))
    y = self.relu(self.conv43(self.pad(y)))
    y = self.relu(self.conv42(self.pad(y)))
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y)))
    return y
    
class Encoder4_2(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder4_2, self).__init__()
    self.fixed = fixed
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1 # 3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2 # 6
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1 # 10
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2 # 13
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1 # 17
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2 # 20
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3 # 23
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4 # 26
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1 # 30
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2 # 33
    )
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.vgg[0])
        load_param(t7_model, 2,  self.vgg[2])
        load_param(t7_model, 5,  self.vgg[5])
        load_param(t7_model, 9,  self.vgg[9])
        load_param(t7_model, 12, self.vgg[12])
        load_param(t7_model, 16, self.vgg[16])
        load_param(t7_model, 19, self.vgg[19])
        load_param(t7_model, 22, self.vgg[22])
        load_param(t7_model, 25, self.vgg[25])
        load_param(t7_model, 29, self.vgg[29])
        load_param(t7_model, 32, self.vgg[32])
      else: # pth model
        net = torch.load(model)
        odict_keys = list(net.keys())
        cnt = 0; i = 0
        for m in self.vgg.children():
          if isinstance(m, nn.Conv2d):
            print("layer %s is loaded with trained params" % i)
            m.weight.data.copy_(net[odict_keys[cnt]]); cnt += 1
            m.bias.data.copy_(net[odict_keys[cnt]]); cnt += 1
          i += 1
  def forward(self, x):
    return self.vgg(x)
  def forward_branch(self, x):
    y_relu1_1 = self.vgg[  : 4](x)
    y_relu2_1 = self.vgg[ 4:11](x)
    y_relu3_1 = self.vgg[11:18](x)
    y_relu4_1 = self.vgg[18:31](x)
    return y_relu1_1, y_relu2_1, y_relu3_1, y_relu4_1

class Decoder4_2(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder4_2, self).__init__()
    self.fixed = fixed
    
    self.conv42 = nn.Conv2d(512,512,3,1,0)
    self.conv41 = nn.Conv2d(512,256,3,1,0)
    self.conv34 = nn.Conv2d(256,256,3,1,0)
    self.conv33 = nn.Conv2d(256,256,3,1,0)
    self.conv32 = nn.Conv2d(256,256,3,1,0)
    self.conv31 = nn.Conv2d(256,128,3,1,0)
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 1,  self.conv51)
        load_param(t7_model, 5,  self.conv44)
        load_param(t7_model, 8,  self.conv43)
        load_param(t7_model, 11, self.conv42)
        load_param(t7_model, 14, self.conv41)
        load_param(t7_model, 18, self.conv34)
        load_param(t7_model, 21, self.conv33)
        load_param(t7_model, 24, self.conv32)
        load_param(t7_model, 27, self.conv31)
        load_param(t7_model, 31, self.conv22)
        load_param(t7_model, 34, self.conv21)
        load_param(t7_model, 38, self.conv12)
        load_param(t7_model, 41, self.conv11)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, y):
    y = self.relu(self.conv42(self.pad(y)))
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y)))
    return y

class SmallEncoder4_2(nn.Module):
  def __init__(self, model=None):
    super(SmallEncoder4_2, self).__init__()
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 16, (3, 3)),
        nn.ReLU(),  # relu1-1 # 3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 16, (3, 3)),
        nn.ReLU(),  # relu1-2 # 6
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 32, (3, 3)),
        nn.ReLU(),  # relu2-1 # 10
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 32, (3, 3)),
        nn.ReLU(),  # relu2-2 # 13
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 64, (3, 3)),
        nn.ReLU(),  # relu3-1 # 17
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-2 # 20
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-3 # 23
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-4 # 26
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu4-1 # 30
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 512, (3, 3)),
        nn.ReLU(),  # relu4-2 # 33
    )
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
  def forward(self, x):
    return self.vgg(x)
    
class Encoder4(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder4, self).__init__()
    self.fixed = fixed
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.vgg[0])
        load_param(t7_model, 2,  self.vgg[2])
        load_param(t7_model, 5,  self.vgg[5])
        load_param(t7_model, 9,  self.vgg[9])
        load_param(t7_model, 12, self.vgg[12])
        load_param(t7_model, 16, self.vgg[16])
        load_param(t7_model, 19, self.vgg[19])
        load_param(t7_model, 22, self.vgg[22])
        load_param(t7_model, 25, self.vgg[25])
        load_param(t7_model, 29, self.vgg[29])
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, x):
    return self.vgg(x)

class SmallEncoder4_16x_plus(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder4_16x_plus, self).__init__()
    self.fixed = fixed
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 16, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 16, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 32, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 32, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 64, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.vgg[0])
        load_param(t7_model, 2,  self.vgg[2])
        load_param(t7_model, 5,  self.vgg[5])
        load_param(t7_model, 9,  self.vgg[9])
        load_param(t7_model, 12, self.vgg[12])
        load_param(t7_model, 16, self.vgg[16])
        load_param(t7_model, 19, self.vgg[19])
        load_param(t7_model, 22, self.vgg[22])
        load_param(t7_model, 25, self.vgg[25])
        load_param(t7_model, 29, self.vgg[29])
      else: # pth model with different arch design but same learnable weights 
        net = torch.load(model)
        odict_keys = list(net.keys())
        cnt = 0; i = 0
        for m in self.vgg.children():
          if isinstance(m, nn.Conv2d):
            print("layer %s is loaded with trained params" % i)
            m.weight.data.copy_(net[odict_keys[cnt]]); cnt += 1
            m.bias.data.copy_(net[odict_keys[cnt]]); cnt += 1
          i += 1
            
  def forward(self, x):
    return self.vgg(x)

class SmallDecoder4_16x(nn.Module):
  def __init__(self):
    super(SmallDecoder4_16x, self).__init__()
    self.conv41 = nn.Conv2d(128, 64,3,1,0)
    self.conv34 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv33 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv32 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv31 = nn.Conv2d( 64, 32,3,1,0, dilation=1)
    self.conv22 = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv21 = nn.Conv2d( 32, 16,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))

  def forward(self, y):
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.conv11(self.pad(y))
    return y

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    
    nn.Upsample(scale_factor=2, mode='nearest'),
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    
    nn.Upsample(scale_factor=2, mode='nearest'),
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    
    nn.Upsample(scale_factor=2, mode='nearest'),
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

# For idea of learning transformation and decoder in one network
class TransformDecoder4(nn.Module):
  def __init__(self):
    super(TransformDecoder4, self).__init__()
    self.conv41 = nn.Conv2d(1024,256,3,1,0)
    self.conv34 = nn.Conv2d(256,256,3,1,0)
    self.conv33 = nn.Conv2d(256,256,3,1,0)
    self.conv32 = nn.Conv2d(256,256,3,1,0)
    self.conv31 = nn.Conv2d(256,128,3,1,0)
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0)
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
  def forward(self, y):
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.conv11(self.pad(y))
    return y

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.decoder2 = TransformDecoder4() # for idea of directly learn transform and decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        
        ## original
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat #wh: stylized feature
        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[-1], t) #?????
        
        ## mine
        '''
        t = torch.cat((content_feat, style_feats[-1]), dim=1)
        g_t = self.decoder2(t) # decoded image
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[-1], content_feat.data)
        '''
        
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s, g_t
