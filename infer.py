# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:15:05 2020

@author: kasy
"""
from argparse import ArgumentParser

import ignite

from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage

from ignite.handlers import ModelCheckpoint, Timer

from ignite.contrib.handlers import ProgressBar

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import RunningAverage, Accuracy, Precision, Recall, Loss, TopKCategoricalAccuracy

from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler




from data import train_loader, eval_train_loader, test_loader
from loss import MultiTaskLoss
import numpy as np

import os
import time

from tqdm import tqdm
import pandas as pd


import torch

import torch.nn as nn

from model import resnext101

from eval import cl_accuracy
from loss import FocalLoss


from itertools import chain

import torch.optim as optim
import torch.nn.functional as F
from data import class_weights
from loss import kappa_loss
from data import e_class_w


from eval import CLAccuracy

model_path = './logging/best.pth'

best_model = resnext101()
best_model.load_state_dict(torch.load(model_path))
best_model = best_model.cuda().eval()

activation = lambda y: y  # no-op
use_regressor = False

def inference_update_with_tta(engine, batch, use_regressor=use_regressor):
    global preds, targets
    best_model.eval()
    with torch.no_grad():
        x, y = batch
        x = x.cuda()
        # Let's compute final prediction as a mean of predictions on x and flipped x
        if use_regressor:
            y_pred1 = best_model(x)[...,-1]
            y_pred2 = best_model(x.flip(dims=(-1, )))[...,-1]
            # calc softmax for submission
            curr_pred = (activation(y_pred1) + activation(y_pred2)) * 0.5
            preds += curr_pred.cpu().squeeze().tolist()
        else:
            y_pred1 = best_model(x)[...,:-1]
            y_pred2 = best_model(x.flip(dims=(-1, )))[...,:-1]
            # calc softmax for submission
            curr_pred = F.softmax(y_pred1, dim=-1) + F.softmax(y_pred2, dim=-1)
            preds += curr_pred.argmax(dim=-1).cpu().squeeze().tolist()
        targets += y.cpu().squeeze().tolist()
        return y_pred1, y

inferencer = Engine(inference_update_with_tta)
ProgressBar(desc="Inference").attach(inferencer)

preds, targets = [], []
result_state = inferencer.run(eval_train_loader, max_epochs=1)
print('valid accuracy:', (np.array(preds) == np.array(targets)).mean())

import scipy as sp

class KappaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.coef = [0.5, 1.5, 2.5, 3.5]
        # define score function:
        self.func = self.quad_kappa
    
    def predict(self, preds):
        return self._predict(self.coef, preds)

    @classmethod
    def _predict(cls, coef, preds):
        if type(preds).__name__ == 'Tensor':
            y_hat = preds.clone().view(-1)
        else:
            y_hat = torch.FloatTensor(preds).view(-1)

        for i,pred in enumerate(y_hat):
            if   pred < coef[0]: y_hat[i] = 0
            elif pred < coef[1]: y_hat[i] = 1
            elif pred < coef[2]: y_hat[i] = 2
            elif pred < coef[3]: y_hat[i] = 3
            else:                y_hat[i] = 4
        return y_hat.int()
    
    def quad_kappa(self, preds, y):
        return self._quad_kappa(self.coef, preds, y)

    @classmethod
    def _quad_kappa(cls, coef, preds, y):
        y_hat = cls._predict(coef, preds)
        return cohen_kappa_score(y, y_hat, weights='quadratic')

    def fit(self, preds, y):
        ''' maximize quad_kappa '''
        print('Early score:', self.quad_kappa(preds, y))
        neg_kappa = lambda coef: -self._quad_kappa(coef, preds, y)
        opt_res = sp.optimize.minimize(neg_kappa, x0=self.coef, method='nelder-mead',
                                       options={'maxiter':100, 'fatol':1e-20, 'xatol':1e-20})
        print(opt_res)
        self.coef = opt_res.x
        print('New score:', self.quad_kappa(preds, y))

    def forward(self, preds, y):
        ''' the pytorch loss function '''
        return torch.tensor(self.quad_kappa(preds, y))

if use_regressor:
    kappa_opt = KappaOptimizer()
    # fit on validation set
    kappa_opt.fit(preds, targets)
    opt_preds = kappa_opt.predict(preds).tolist()

    _ = pd.DataFrame(preds).hist()
    _ = pd.DataFrame(opt_preds).hist()
    
preds, targets = [], []
result_state = inferencer.run(test_loader, max_epochs=1)

if use_regressor:
    preds = kappa_opt.predict(preds).tolist()
    
submission = pd.DataFrame({'id_code': pd.read_csv('./data/sample_submission.csv').id_code.values,
                         'diagnosis': np.squeeze(preds).astype(np.int32)})

#submission.hist()
#submission.head()

submission_file = 'V3.csv'

submission.to_csv(submission_file, index=False)