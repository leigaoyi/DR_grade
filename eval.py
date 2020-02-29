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
    
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class CLAccuracy(Metric):

    def __init__(self, ignored_class=None, output_transform=lambda x: x, device=None):
        self.ignored_class = ignored_class
        self._num_correct = None
        self._num_examples = None
        self.Acc = 0
        super(CLAccuracy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(CLAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        self.Acc = accuracy_score(torch.argmax(y_pred[...,:-1], dim=1).cpu().numpy(),
                          y.cpu().numpy())
#        indices = torch.argmax(y_pred, dim=1)
#
#        mask = (y != self.ignored_class)
#        mask &= (indices != self.ignored_class)
#        y = y[mask]
#        indices = indices[mask]
#        correct = torch.eq(indices, y).view(-1)
#
#        self._num_correct += torch.sum(correct).item()
#        self._num_examples += correct.shape[0]



    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
#        if self._num_examples == 0:
#            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
#        return self._num_correct / self._num_examples    
        return self.Acc
    