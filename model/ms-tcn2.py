import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channel, out_channel) -> None:
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channel, out_channel, 3, padding=dilation, dilation=dilation)
        self.conv1x1 = nn.Conv1d(in_channel, out_channel, 1)
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv1x1(out)
        out = self.dropout(out)
        return x + out

