import pdb 
import os

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

def RRL(dataset, args, model, train=True): 
    
    class _RRL(dataset): 
        def __init__(self, args, model, train=True):
            super(_RRL, self).__init__(args, train)
            self.model_ref = model

        def __getitem__(self, idx): 
            lr_tensor, hr_tensor, filename = super(_RRL, self).__getitem__(idx)
            pred = self.model_ref.forward(lr_tensor.unsqueeze_(0),0)

            featmaps = self.model_ref.model.tail.modules().next()._modules['0'].outputs
            residual = hr_tensor - pred[0]

            return featmaps[0][0], residual, filename

    return _RRL(args, model, train)