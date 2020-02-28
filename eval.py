# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:08:34 2020

@author: kasy
"""

import torch

from sklearn.metrics import cohen_kappa_score, accuracy_score

def qw_kappa(pred, y):  ## quadratic weights
    return cohen_kappa_score(torch.argmax(pred[...,:-1], dim=1).cpu().numpy(),
                             y.cpu().numpy(),
                             weights='quadratic')

def cl_accuracy(pred, y):
    return accuracy_score(torch.argmax(pred[...,:-1], dim=1).cpu().numpy(),
                          y.cpu().numpy())
    
