import torch.nn as nn 
import numpy as np 
import torch
import pdb

class SSIM(nn.Module): 

    def __init__(self, width, batch_size, n_channel, cuda, c1=.01**2, c2=.02**2, sigma=5.): 
        super(SSIM, self).__init__()
        self.c1 = c1 
        self.c2 = c2 
        self.sigma = sigma 

        self.w = np.exp(-1.*np.arange(-(width/2), width/2)**2/(2*self.sigma**2))
        self.w = np.outer(self.w, self.w.reshape((width, 1)))
        self.w = self.w/np.sum(self.w)

        temp = np.zeros(shape=(n_channel,n_channel,width,width),dtype=np.float32)
        l = range(n_channel)
        temp[l,l] = self.w
        self.w = temp

        self.w = torch.Tensor(self.w)

        if cuda: 
            self.w = self.w.cuda()
    
    def forward(self, input, target): 
        nb, nc = input.shape[0], input.shape[1]

        mux = nn.functional.conv2d(input, self.w)
        muy = nn.functional.conv2d(target, self.w)
        sigmax2 = nn.functional.conv2d(input**2,self.w) - mux**2
        sigmay2 = nn.functional.conv2d(target**2,self.w) - muy**2
        sigmaxy = nn.functional.conv2d(input*target,self.w) - mux*muy

        l = (2 * mux * muy + self.c1) / (mux ** 2 + muy ** 2 + self.c1)
        cs = (2 * sigmaxy + self.c2) / (sigmax2 + sigmay2 + self.c2)

        out = 1 - torch.sum(l * cs) / (nb * nc)

        return out