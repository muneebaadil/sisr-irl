import torch
import torch.nn as nn
import numpy as np
import math
from model import common

def make_model(args, parent=False): 
    return DenseSkip(args)

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
        self.add_module('conv{}'.format(1), conv_layer)
        self.conv_layers.append(conv_layer)

        next_in_chans = growth_rate
        for i in xrange(n_layers-1): 
            conv_layer = conv(next_in_chans, growth_rate, kernel_size)
            self.add_module('conv{}'.format(i+2), conv_layer)
            self.conv_layers.append(conv_layer)

            next_in_chans += growth_rate
        
    def forward(self, x):

        outs = []
        for conv in self.conv_layers: 
            out = self.act(conv(x))
            outs.append(out)
            x = torch.cat(outs, 1)

        return x

class DenseSkip(nn.Module): 
    def __init__(self, args, conv=common.default_conv): 
        super(DenseSkip, self).__init__()

        self.act = nn.ReLU(True)
        kernel_size = 3 
        growth_rate = args.growth_rate
        channels = 128
        n_denseblocks = args.n_denseblocks
        scale = args.scale[0]
        self.is_sub_mean = args.is_sub_mean

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Conv2d(in_channels=args.n_channel_in, out_channels=channels,
                             kernel_size=kernel_size,padding=1)

        self.dense_blocks = []
        for i in xrange(n_denseblocks): 
            db = _Dense_Block(growth_rate, channels, args.n_layers)
            self.add_module('db{}'.format(i+1), db)
            self.dense_blocks.append(db)

        self.bottleneck = nn.Conv2d(in_channels=channels*(n_denseblocks+1), 
                                    out_channels=channels*2, kernel_size=1,
                                     stride=1, padding=0, bias=False)

        self.tail = [common.Upsampler(nn.ConvTranspose2d, scale=scale, n_feat=channels*2,
                     act=self.act, bias=False, type='deconv')]
        self.tail = nn.Sequential(*self.tail)

        self.reconstruction = nn.Conv2d(in_channels=channels*2, out_channels=args.n_channel_out,
                                         kernel_size=kernel_size, stride=1, padding=1, bias=False)

        #weight initialization
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

    def forward(self, x): 
        if self.is_sub_mean: 
            x = self.sub_mean(x)

        x = self.act(self.head(x))
        outs = [x]

        for db in self.dense_blocks: 
            x = db(x)
            outs.append(x)

        x = torch.cat(outs, 1)
        x = self.bottleneck(x)        
        x = self.tail(x)
        x = self.reconstruction(x)

        if self.is_sub_mean: 
            x = self.add_mean(x)

        return x