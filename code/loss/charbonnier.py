import torch 
import torch.nn as nn

class Charbonnier(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(Charbonnier, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error)
        return loss 