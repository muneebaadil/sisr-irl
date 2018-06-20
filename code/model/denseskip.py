import torch
import torch.nn as nn
import numpy as np
import math
from model import common

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class _Dense_Block(nn.Module):
    def __init__(self, growth_rate, in_channels, n_layers, conv=common.default_conv):
        super(_Dense_Block, self).__init__()
        kernel_size = 3 
        self.act = nn.ReLU(True)

        self.conv_layers = []
        conv_layer = conv(in_channels, growth_rate, kernel_size)
        self.conv_layers.append(conv_layer)

        next_in_chans = growth_rate
        for _ in xrange(n_layers-1): 
            conv_layer = conv(next_in_chans, growth_rate, kernel_size)
            self.conv_layers.append(conv_layer)

            next_in_chans += growth_rate
        
    def forward(self, x):

        outs = []
        for conv in self.conv_layers: 
            out = self.act(conv(x))
            outs.append(out)
            x = torch.cat(outs, 1)

        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.relu = nn.PReLU()
        self.lowlevel = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv2d(in_channels=1152, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.reconstruction = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.denseblock1 = self.make_layer(_Dense_Block, 128)
        self.denseblock2 = self.make_layer(_Dense_Block, 256)
        self.denseblock3 = self.make_layer(_Dense_Block, 384)
        self.denseblock4 = self.make_layer(_Dense_Block, 512)
        self.denseblock5 = self.make_layer(_Dense_Block, 640)
        self.denseblock6 = self.make_layer(_Dense_Block, 768)
        self.denseblock7 = self.make_layer(_Dense_Block, 896)
        self.denseblock8 = self.make_layer(_Dense_Block, 1024)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):    
        residual = self.relu(self.lowlevel(x))

        out = self.denseblock1(residual)
        concat = torch.cat([residual,out], 1)

        out = self.denseblock2(concat)
        concat = torch.cat([concat,out], 1)

        out = self.denseblock3(concat)
        concat = torch.cat([concat,out], 1)
        
        out = self.denseblock4(concat)
        concat = torch.cat([concat,out], 1)
        
        out = self.denseblock5(concat)
        concat = torch.cat([concat,out], 1)
        
        out = self.denseblock6(concat)
        concat = torch.cat([concat,out], 1)
        
        out = self.denseblock7(concat)
        concat = torch.cat([concat,out], 1)
        
        out = self.denseblock8(concat)
        out = torch.cat([concat,out], 1)

        out = self.bottleneck(out)

        out = self.deconv(out)

        out = self.reconstruction(out)
       
        return out
        
class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        return loss 