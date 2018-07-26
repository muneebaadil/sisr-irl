import common 
import torch.nn as nn 

def make_model(args): 
    return SRResNet(args)

class SRResNet(nn.Module): 
    def __init__(self, args, conv=common.default_conv): 
        super(SRResNet, self).__init__()
        
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.LeakyReLU(negative_slope=0.2)
        self.is_sub_mean = args.is_sub_mean
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        head = [conv(args.n_channel_in, args.n_feats, kernel_size),
                nn.LeakyReLU(negative_slope=0.2)]
        body = [common.ResBlock(conv,args.n_feats,kernel_size,bn=True,act=act) \
                for _ in xrange(args.n_resblocks)]
        body.extend([conv(args.n_feats, args.n_feats, kernel_size), 
                    nn.BatchNorm2d(args.n_feats)])
        tail = [
            common.Upsampler(conv, scale, args.n_feats, act=nn.LeakyReLU),
            conv(args.n_feats, args.n_channel_out, kernel_size)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x): 
        if self.is_sub_mean: 
            x = self.sub_mean(x)

        x = self.head(x)

        res = self.body(x)
        res += x

        self.down_feats = res
        x = self.tail(res)

        if self.is_sub_mean: 
            x = self.sub_mean(x)

        return x