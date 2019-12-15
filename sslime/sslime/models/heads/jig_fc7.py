#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch.nn as nn

from sslime.core.config import config as cfg


class JIG_FC7(nn.Module):
    def __init__(self, dims):
        super(JIG_FC7, self).__init__()
        
        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7_s1',nn.Linear(dims[0], dims[1]))
        self.fc7.add_module('relu7_s1',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7_s1',nn.Dropout(p=0.5))


    
    def forward(self, x):
        print("FC7 SHAPE: " , x.size())
        
        return  self.fc7(x.view(1,-1))
    
