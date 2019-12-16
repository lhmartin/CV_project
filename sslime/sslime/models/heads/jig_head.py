#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch.nn as nn

from sslime.core.config import config as cfg
from torch import cat


class JIG_HEAD(nn.Module):
    def __init__(self, dims):
        super(JIG_HEAD, self).__init__()
        
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(dims[0], dims[1]))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

 
        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7_s1',nn.Linear(dims[2], dims[3]))
        self.fc7.add_module('relu7_s1',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7_s1',nn.Dropout(p=0.5))
        
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier',nn.Linear(dims[4], dims[5]))

    
    def forward(self, x):
        B = 1
        x = x.transpose(0,1)
        x = x.transpose(1,2)
        x_list = []
        for i in range(3):
            for j in range(3):
                z = self.fc6(x[i][j])
                x_list.append(z)

        x = cat(x_list,0)
        x = self.fc7(x.view(1,-1))
        x = self.classifier(x)
        return x
    
