import torch.nn as nn 
import numpy as np 
import torch

class SSIM(nn.Module): 

    def __init__(self, width, batch_size, c1=.01, c2=.02, sigma=5.): 
        super(SSIM, self).__init__()
        self.c1 = c1 
        self.c2 = c2 
        self.sigma = sigma 

        self.w = np.exp(-1.*np.arange(-(width/2), width/2)**2/(2*self.sigma**2))
        self.w = np.outer(self.w, self.w.reshape((width, 1)))
        self.w = self.w/np.sum(self.w)
        self.w = np.reshape(self.w, (1, 1, width, width))
        self.w = np.tile(self.w, (batch_size, 3, 1, 1))

        self.w = torch.Tensor(self.w)
    
    def forward(self, input, target): 
        nb, nc = input.shape[0], input.shape[1]

        mux = torch.sum((self.w * input).view(nb,nc,-1), dim=-1, keepdim=True)
        muy = torch.sum((self.w * target).view(nb,nc,-1), dim=-1, keepdim=True)
        sigmax2 = torch.sum((self.w * input ** 2).view(nb,nc,-1), dim=-1, keepdim=True) - mux**2
        sigmay2 = torch.sum((self.w * target ** 2).view(nb,nc,-1), dim=-1, keepdim=True) - muy**2
        sigmaxy = torch.sum((self.w * input * target).view(nb,nc,-1), dim=-1, keepdim=True) - mux * muy

        l = (2 * mux * muy + self.c1) / (mux ** 2 + muy ** 2 + self.c1)
        cs = (2 * sigmaxy + self.c2) / (sigmax2 + sigmay2 + self.c2)

        out = 1 - torch.sum(l * cs) / (nb * nc)

        return out