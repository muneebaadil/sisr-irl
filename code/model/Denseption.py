from model import common

import torch.nn as nn
from torch import cat

def make_model(args, parent=False): 
    return DenseNetAll(args)

class DenseNetAll(nn.Module):

    def __init__(self, args, conv=common.default_conv):
        super(DenseNetAll, self).__init__()

        n_denseblocks = args.n_denseblocks
        n_layers = args.n_layers
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, -1)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        #define the body
        modules_body = []
        for i in xrange(n_denseblocks): 
            modules_body.append(common.DenseBlock(n_layers, n_feats, conv, kernel_size, 
            act=act, res_scale=args.res_scale))
        
        # define tail module
        modules_tail = [
            conv(n_feats*(n_denseblocks+1), n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        return 

    def forward(self, x): 
        x = self.sub_mean(x)
        x = self.head(x)

        featmaps = self.body(x)

        out = self.tail(featmaps)
        out = self.add_mean(out)
        
        return out

    def cuda(self, device=None): 
        super(DenseNetAll, self).cuda()
        
        for m in self.body: 
            m.cuda() 
        return 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                            'whose dimensions in the model are {} and '
                                            'whose dimensions in the checkpoint are {}.'
                                            .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                    .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))