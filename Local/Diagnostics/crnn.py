"""
Model architecture of modulation tensorgram-convoltional recurrent network (MTR-CRNN).
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as scio
import sys, os, glob
from collections import Counter

import torch as T
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset
from tqdm import tqdm

# %% CRNN
class crnn_cov_3d(nn.Module):
    
    def __init__(self, num_class:int, 
                 msr_size:tuple, 
                 rnn_hidden_size:int, 
                 dropout:float,
                 tem_fac:list):
        
        super(crnn_cov_3d, self).__init__()
        
        self.num_class = num_class
        self.rnn_hidden_size = rnn_hidden_size
        self.dp = nn.Dropout(p=dropout)
        self.num_freq = msr_size[0]
        self.num_mod = msr_size[1]
        # self.dp_3d = nn.Dropout3d(p=dropout)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.tem_fac = tem_fac # temporal pooling factors for each cnn block. E.g., [2,3,1]
        
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1,4,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(4),
            nn.MaxPool3d((self.tem_fac[0],1,1)),
            self.relu
            )
        
        self.cnn2 = nn.Sequential(
            nn.Conv3d(4,16,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(16),
            nn.MaxPool3d((self.tem_fac[1],1,1)),
            self.relu
            )
        
        self.cnn3 = nn.Sequential(
            nn.Conv3d(16,4,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(4),
            nn.MaxPool3d((self.tem_fac[2],1,1)),
            self.relu
            )
        
                
        # self.cnn4 = nn.Sequential(
        #     nn.Conv3d(4,1,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
        #     nn.BatchNorm3d(1),
        #     nn.MaxPool3d((21,1,1)),
        #     self.relu
        #     )5
        
        self.downsample = nn.MaxPool3d((2,2,2))
                
        
        self.CNNblock = nn.Sequential(
            self.cnn1,
            self.cnn2,
            self.cnn3
            )
        
        # self.Att = CBAM_Att(channel=4)
        
        self.fc1 = nn.Sequential(
            nn.Linear(4*self.num_freq*self.num_mod, 128),
            nn.BatchNorm1d(128),
            self.relu,
            self.dp
            )
        
        # RNN
        self.rnn1 = nn.GRU(input_size=128, 
                            hidden_size=self.rnn_hidden_size,
                            num_layers=3,
                            bidirectional=True, 
                            batch_first=True)
        
        self.layer_norm = nn.LayerNorm([2*self.rnn_hidden_size,int(150/np.product(self.tem_fac))])
        self.maxpool = nn.MaxPool1d(int(150/np.product(self.tem_fac)))
        
        # self.fc2 = nn.Sequential(
        #     nn.Linear(512,64),
        #     nn.BatchNorm1d(64),
        #     self.relu,
        #     nn.Linear(64,1)
        #     )
        
        self.fc2 = nn.Linear(self.rnn_hidden_size*2,self.num_class)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self,x):

        # 3D-CNN block
        ot = self.CNNblock(x)
        # print(ot.size())
        
        # Attention CBAM
        # ot = self.Att(ot)
        # print(ot.size())
        
        # flatten
        ot = torch.permute(ot,(0,2,1,3,4))
        time_step = ot.size(1)
        # print(ot.size())
        
        ot = ot.reshape((ot.size(0) * time_step,-1))
        # print(ot.size())
        
        # fc layer
        ot =self.fc1(ot)
        ot = ot.reshape((x.size(0),time_step,-1))
        # print(ot.size())
        
        # RNN block
        ot, _ = self.rnn1(ot)
        ot = torch.permute(ot,(0,2,1))
        ot = self.layer_norm(ot)
        ot = self.maxpool(ot)
        # print(ot.size())        
           
        # fc layer
        ot = ot.reshape((ot.size(0),-1))
        ot = self.fc2(ot)
        # print(ot.size())
        
        return ot


if __name__ == '__main__':

    toy_input = torch.randn(16,1,150,23,8)
    model = crnn_cov_3d(num_class=1,msr_size=(23,8),rnn_hidden_size=128,dropout=0.7,tem_fac=[2,3,1])
    toy_output = model(toy_input)
    print(toy_output)