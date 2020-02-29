# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:04:02 2020

@author: kasy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        self.alpha = self.alpha.type(inputs.type(), non_blocking=True) # fix type and device

        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha[targets] * (1-pt)**self.gamma * CE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()
        return F_loss
    
# activation = lambda y: (n_classes-1) * torch.sigmoid(y)
# activation = lambda y: (n_classes-1) * (0.5 + 0.5 * y / (1 + y.abs()))  # linear sigmoid
activation = lambda y: y  # no-op

def cont_kappa(input, targets, activation=None):
    ''' continuos version of quadratic weighted kappa '''
    n = len(targets)
    y = targets.float().unsqueeze(0)
    pred = input.float().squeeze(-1).unsqueeze(0)
    if activation is not None:
        pred = activation(pred)
    wo = (pred - y)**2
    we = (pred - y.t())**2
    return 1 - (n * wo.sum() / we.sum())
# adapted from keras version: https://www.kaggle.com/ryomiyazaki/keras-simple-implementation-of-qwk-for-regressor
    
kappa_loss = lambda pred, y: 1 - cont_kappa(pred, y)  # from 0 to 2 instead of 1 to -1

# balance between metric optimisation and classification accuracy
class MultiTaskLoss(FocalLoss):
    def __init__(self, alpha=None, gamma=2.0, second_loss=F.mse_loss, second_mult=0.1):
        super().__init__(alpha, gamma)
        self.second_loss = second_loss
        self.second_mult = second_mult

    def forward(self, inputs, targets):
        loss  = super().forward(inputs[...,:-1], targets)  # focal loss
        loss += self.second_mult * self.second_loss(inputs[...,-1], targets.float())
        return loss
    
