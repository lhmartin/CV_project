#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch.nn as nn

from sslime.core.config import config as cfg


class JIG_CLASS(nn.Module):
    def __init__(self, dims):
        super(JIG_CLASS, self).__init__()
        
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier',nn.Linear(dims[0], dims[1]))


    
    def forward(self, x):
        print("CLASSIFIER: ")
        print(x.size())
        B,T,C,H,W = x.size()
        
        return self.classifier(x)
    
