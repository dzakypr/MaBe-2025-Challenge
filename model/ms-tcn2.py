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

class SingleStageModel(nn.Module):
    # 2,4,8,16,32,64,128,512,1024,2048,4096
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        
        self.layers = nn.ModuleList([
            DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        
        self.stages = nn.ModuleList([
            SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)
            for _ in range(num_stages - 1)
        ])

    def forward(self, x, mask):
        out = self.stage1(x)
        
        outputs = [out]

        for s in self.stages:
            in_refinement = F.softmax(out, dim=1) * mask[:, 0:1, :]
            out = s(in_refinement)
            outputs.append(out)

        return outputs


class TMSELoss(nn.Module):
    def __init__(self, threshold=4.0):
        super(TMSELoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, predictions, batch_mask):
        loss = self.mse(torch.log_softmax(predictions[:, :, 1:], dim=1), 
                        torch.log_softmax(predictions[:, :, :-1], dim=1))

        loss = torch.clamp(loss, min=0, max=self.threshold**2)
        
        mask = batch_mask[:, :, 1:]
        loss = torch.mean(loss * mask)
        
        return loss