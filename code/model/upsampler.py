from model import common 
import torch.nn as nn 

def make_model(args): 
    return Upsampler(args)

class Upsampler(nn.Module): 
    def __init__(self, args, conv=common.default_conv): 
        super(Upsampler, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        kernel_size = 3 
        scale = args.scale[0]

        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        body, tail = [], []
        for i in xrange(args.n_layers): 
            body.extend([conv(args.n_channel_in if (i==0) else args.n_feats,
             args.n_feats, kernel_size), nn.ReLU(True)])

        tail.extend([common.Upsampler(conv, scale, args.n_feats, act=False),
                conv(args.n_feats, args.n_channel_out, kernel_size)])
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x): 
        x = self.sub_mean(x)
        self.down_feats = self.body(x)
        x = self.tail(self.down_feats)
        x = self.add_mean(x)
        return x