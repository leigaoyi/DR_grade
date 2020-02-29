# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:58:28 2020

@author: kasy
"""

from model import EfficientNet

from data import train_loader
from loss import MultiTaskLoss
import numpy as np

import os
import time

from tqdm import tqdm

import torch

import torch.nn as nn

from model import resnext50

from eval import cl_accuracy
from loss import FocalLoss

use_amp = False

if use_amp:
    from apex import amp


ckpt_dir = './checkpoints/'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

model = resnext50()

#model = EfficientNet(num_classes=1000, 
#                     width_coefficient=1.4, depth_coefficient=1.8,
#                     dropout_rate=0.4)

#from collections import OrderedDict
#
#model_state = torch.load("./pretrained/efficientnet-b4-6ed6700e.pth")
#
## A basic remapping is required
#mapping = {
#    k: v for k, v in zip(model_state.keys(), model.state_dict().keys())
#}
#mapped_model_state = OrderedDict([
#    (mapping[k], v) for k, v in model_state.items()
#])
#
#model.load_state_dict(mapped_model_state, strict=False)
#
#in_features, out_features = model.head[6].in_features, model.head[6].out_features
#
#n_classes = 5
#model.head[6] = nn.Linear(in_features, n_classes+1) # classification +  kappa regressor


assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
torch.backends.cudnn.benchmark = True

device = "cuda"

model = model.to(device)

from itertools import chain

import torch.optim as optim
import torch.nn.functional as F
from data import class_weights
from loss import kappa_loss
from data import e_class_w

criterion_g = MultiTaskLoss(gamma=2., alpha=class_weights, second_loss=kappa_loss, second_mult=0.5)
criterion_e = FocalLoss(alpha=e_class_w)

lr = 1e-2  # placeholder only! check the LR schedulers below

#optimizer = optim.SGD([
#    {
#        "params": chain(model.stem.parameters(), model.blocks.parameters()),
#        "lr": lr * 0.1,
#    },
#    {
#        "params": model.head[:6].parameters(),
#        "lr": lr * 0.2,
#    },    
#    {
#        "params": model.head[6].parameters(), 
#        "lr": lr
#    }], 
#    momentum=0.99, weight_decay=1e-4, nesterov=True)

optimizer = optim.SGD(model.parameters(),lr=lr, momentum=0.99, weight_decay=1e-4, nesterov=True)

for param in model.parameters():
    param.requires_grad = True


if use_amp:
    # Initialize Amp
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", num_losses=1)
    
epoch_num = 1000
for epoch in range(epoch_num):
    model.train()
    
#    if epoch == 0:
#        for name, child in model.named_children():
#            if name == 'head':
#                #pbar.log_message('training {}'.format(name))
#                for param in child.parameters():
#                    param.requires_grad = True
#            else:
##                 pbar.log_message(f'"{name}" is frozen')
#                for param in child.parameters():
#                    param.requires_grad = False
#    else:
#        #pbar.log_message("Epoch {}: training all layers".format(epoch))
#        for name, child in model.named_children():
#            for param in child.parameters():
#                param.requires_grad = True
    acc_list = []
    loss_list = []
    
    start_t = time.time()
    for x, y, z in train_loader:
        #print(y)
        
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        
        y_pred, z_pred = model(x)
    
        # Compute loss 
        loss_g = criterion_g(y_pred, y)
        loss_e = criterion_e(z_pred, z)
        loss = loss_g + loss_e
        
        acc = cl_accuracy(y_pred, y)
        acc_list.append(acc)
        loss_list.append(loss.item())
    
        optimizer.zero_grad()
        if use_amp:
            with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        #print('Output ', y_pred.max().item())
    end_t = time.time()
    cost_t = -(start_t - end_t)/60
    mean_acc = np.mean(acc_list)
    print('Epoch {2} Loss {0:.4f}, acc {1:.4f}, time {3:.4f} min'.format(np.mean(loss_list), mean_acc,\
          epoch, cost_t))
    
    with open('Accuracy.txt', 'a+') as f:
        f.write('Epoch {0} Acc {1:.4f} \n'.format(epoch+1, mean_acc))
               
    if (epoch+1)%50 == 0:
        path = os.path.join(ckpt_dir, str(epoch+1)+'_{:.4f}.pt'.format(mean_acc))
        torch.save(model.state_dict(), path)
        
    