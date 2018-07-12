import torch 
import torch.nn as nn 

class WeightedMSE(nn.Module):
    def __init__(self):
        pass 
    
    def forward(self, input, target): 
        losses = (input-target)**2
        weights = torch.abs(target) / torch.sum(torch.abs(target))
        loss = torch.sum(weights*losses)
        return loss