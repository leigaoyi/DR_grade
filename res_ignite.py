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

#from ignite.handlers import ModelCheckpoint, Timer
from utils_ckpt import ModelCheckpoint

from ignite.contrib.handlers import ProgressBar
#=======================import ignite=================

from data import train_loader, eval_train_loader
from loss import MultiTaskLoss
import numpy as np

import os
import time

from tqdm import tqdm

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
#model = resnext101()

def get_data_loaders(train_batch_size, val_batch_size):

    return train_loader, eval_train_loader

criterion = MultiTaskLoss(gamma=2., alpha=class_weights, second_loss=kappa_loss, second_mult=0.5)

output_dir = './logging/'
model_name = 'Res101'
#CKPT_PREFIX = ""


def run(train_batch_size, val_batch_size, epochs, lr, momentum, display_gpu_info):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = resnext101()
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
#    evaluator = create_supervised_evaluator(
#        model, metrics={"accuracy": Accuracy(), "nll": Loss(criterion)}, device=device
#    )
    
    evaluator = create_supervised_evaluator(
        model, metrics={"Accuracy": Loss(cl_accuracy), "nll": Loss(criterion)}, device=device
    )
    

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    if display_gpu_info:
        from ignite.contrib.metrics import GpuInfo

        GpuInfo().attach(trainer, name="gpu")

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

#    @trainer.on(Events.EPOCH_COMPLETED)
#    def log_training_results(engine):
#        evaluator.run(train_loader)
#        metrics = evaluator.state.metrics
#        #evaluator.state.metrics["Accuracy"] = round(evaluator.state.metrics["Accuracy"], 4)
#        avg_accuracy = metrics["Accuracy"]
#        avg_nll = metrics["nll"]
#        
#        with open('acc_train.txt', 'a+') as f:
#            f.write('Epoch {0} Acc {1:.4f} \n'.format(engine.state.epoch, avg_accuracy))
#        
#        pbar.log_message(
#            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
#                engine.state.epoch, avg_accuracy, avg_nll
#            )
#        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        
        avg_accuracy = metrics["Accuracy"]
        
        with open('acc_train.txt', 'a+') as f:
            f.write('Epoch {0} Acc {1:.4f} \n'.format(engine.state.epoch, avg_accuracy))
                
        avg_nll = metrics["nll"]
        pbar.log_message(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                engine.state.epoch, avg_accuracy, avg_nll
            )
        )

        pbar.n = pbar.last_print_n = 0
        
    def default_score_fn(engine):
        #score = engine.state.metrics['ClKappa']
        score = engine.state.metrics['Accuracy']
        return score 
    
#    checkpoint_handler = ModelCheckpoint(output_dir, CKPT_PREFIX, n_saved=10, \
#                                     require_empty=False)    
    
    
#    trainer.add_event_handler(
#        event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={"ResNet101": model}
#    )
        
    best_model_handler = ModelCheckpoint(dirname=output_dir,
                                     filename_prefix="best",                                     
                                     score_name="ACC",
                                     score_function=default_score_fn,
                                     n_saved=4,
                                     require_empty=False,
                                     atomic = False,
                                     create_dir= True)
    evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=best_model_handler, to_save={"".format(model_name): model})
    
    # Clear cuda cache between training/testing
    @trainer.on(Events.EPOCH_COMPLETED)
    @evaluator.on(Events.COMPLETED)
    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()
            
    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="input batch size for training (default: 64)")
    parser.add_argument(
        "--val_batch_size", type=int, default=100, help="input batch size for validation (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument(
        "--display_gpu_info",
        default = True,
        action="store_true",
        help="Display gpu usage info. This needs python 3.X and pynvml package",
    )

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.display_gpu_info)
    
    