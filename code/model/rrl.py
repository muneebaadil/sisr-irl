from model import common 
import torch.nn as nn 
import pdb
import model
from importlib import import_module

def make_model(args, ckp, parents=False): 
    return RRL(args, ckp)

class RRL(nn.Module): 
    def __init__(self, args, ckp, conv=common.default_conv): 
        super(RRL, self).__init__()

        args_ = args

        module = import_module('model.' + args.model.lower())
        self.master_branch = module.make_model(args)

        if not args.train_jointly: 
            for param in self.master_branch.parameters(): 
                param.requires_grad = False
        
        args_.n_channel_in = args.n_feats
        args_.scale = [1]
        if args.half_feats: 
            args_.n_feats = args.n_feats / 2 
        if args.half_resblocks: 
            args_.n_resblocks = args.n_resblocks / 2
        args_.is_sub_mean = False

        self.recons_branch = module.make_model(args_)

    def forward(self, x): 
        y1 = self.master_branch(x)
        y2 = self.recons_branch(self.master_branch.featmaps)

        out = y1 + y2

        return out

    def load_state_dict(self, state_dict, strict=True): 
        self.master_branch.load_state_dict(state_dict, strict)