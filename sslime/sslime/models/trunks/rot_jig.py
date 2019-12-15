import torch.nn as nn
from sslime.models.Layers import LRN
from sslime.utils.utils import Flatten, parse_out_keys_arg
from torch import cat


class ROT_JIG(nn.Module):
    def __init__(self,classes=1000):
        super(ROT_JIG, self).__init__()
     
        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1',nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('relu1_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv.add_module('relu2_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('relu4_s1',nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('relu5_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1',nn.MaxPool2d(kernel_size=3, stride=2))

        #self.fc6 = nn.Sequential()
        #self.fc6.add_module('fc6_s1',nn.Linear(256*9, 1024))
        #self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        #self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))
#
        #self.fc7 = nn.Sequential()
        #self.fc7.add_module('fc7',nn.Linear(9*1024,4096))
        #self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        #self.fc7.add_module('drop7',nn.Dropout(p=0.5))
#
        #self.classifier = nn.Sequential()
        #self.classifier.add_module('fc8',nn.Linear(4096, classes))

   
    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.conv(x[i])
            
            x_list.append(z)

        x = cat(x_list,1)
        print(x.shape)

        return x
    
  