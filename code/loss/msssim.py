import torch.nn as nn 
import numpy as np 
import torch 
from functools import reduce

class MSSSIM(nn.Module): 
    def __init__(self, width, batch_size, n_channel,
                 cuda, c1=.01**2, c2=.02**2, n_sigmas=5):
        super(MSSSIM, self).__init__()

        self.c1 = c1 
        self.c2 = c2 
        sigmas = [0.5 * 2 ** i for i in xrange(n_sigmas)]

        self.weights = np.zeros(shape=(n_sigmas,n_channel,n_channel,width,width),
                                dtype=np.float32)

        def _get_kernel(sigma): 
            w = np.exp(-1.*np.arange(-(width/2), width/2)**2/(2*sigmas[n_layer]**2))
            w = np.outer(w, w.reshape((width, 1)))
            w = w/np.sum(w)
            out = np.zeros(shape=(n_channel, n_channel, width, width), 
                            dtype=np.float32)
            out[range(n_channel),range(n_channel)] = w
            return out

        for n_layer in xrange(n_sigmas):
            self.weights[n_layer] = _get_kernel(sigmas[n_layer])
        
        self.weights = torch.Tensor(self.weights)

        if cuda: 
            self.weights = self.weights.cuda()
        
    def forward(self, input, target): 
        def _forward(kernel): 
            mux = nn.functional.conv2d(input, kernel)
            muy = nn.functional.conv2d(target, kernel)
            sigmax2 = nn.functional.conv2d(input**2,kernel) - mux**2
            sigmay2 = nn.functional.conv2d(target**2,kernel) - muy**2
            sigmaxy = nn.functional.conv2d(input*target,kernel) - mux*muy

            return mux,muy,sigmax2,sigmay2,sigmaxy
            
        nb, nc = input.shape[0], input.shape[1]

        cs = []
        for weight in self.weights: 
            mux,muy,sigmax2,sigmay2,sigmaxy = _forward(weight)
            _cs = (2 * sigmaxy + self.c2) / (sigmax2 + sigmay2 + self.c2)
            cs.append(_cs)

        pcs = reduce(lambda x,y:x*y, cs)
        l = (2 * mux * muy + self.c1)/(mux ** 2 + muy **2 + self.c1)
        
        out = 1 - torch.sum((l*pcs) / (nb*nc))
        return out