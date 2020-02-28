# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:58:28 2020

@author: kasy
"""

from model import EfficientNet

from data import train_loader, test_loader, eval_train_loader
from loss import MultiTaskLoss

import torch
#from apex import amp
import torch.nn as nn

from model import resnext50

from eval import cl_accuracy

use_amp = False

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

criterion = MultiTaskLoss(gamma=2., alpha=class_weights, second_loss=kappa_loss, second_mult=0.5)
lr = 1e-3  # placeholder only! check the LR schedulers below

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
    

for epoch in range(200):
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
    
    for x, y in train_loader:
        #print(y)
        
        x = x.to(device)
        y = y.to(device)
        
        y_pred = model(x)
    
        # Compute loss 
        loss = criterion(y_pred, y)
        
        acc = cl_accuracy(y_pred, y)
    
        optimizer.zero_grad()
        if use_amp:
            with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        #print('Output ', y_pred.max().item())
        print('Loss {0:.4f}, acc {1:.4f}'.format(loss.item(), acc))
        
    
    