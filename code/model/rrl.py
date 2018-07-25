from model import common 
import torch.nn as nn 
import model
from importlib import import_module
from copy import deepcopy
from functools import reduce 

def make_model(args, ckp, parents=False): 
    return RRL(args, ckp)

class RRL(nn.Module): 
    def __init__(self, args, ckp, conv=common.default_conv): 
        super(RRL, self).__init__()

        args_ = deepcopy(args)
        self.auto_feats = True if (args.model.lower() == 'lapsrn') else False
        self.branch_label = args.branch_label.lower()
        self.down_feats = args.down_feats

        self.branches = []  
        module = import_module('model.upsampler')
        self.master_branch = module.make_model(args)

        if not args.train_jointly: 
            for param in self.master_branch.parameters(): 
                param.requires_grad = False

        for i in xrange(args.n_branches):

            args_.n_channel_in = args.n_feats

            if not args.down_feats: 
                args_.scale = [args_.scale[0] // 2] 

            if args.half_feats: 
                args_.n_feats = args_.n_feats / 2 
            if args.half_resblocks: 
                args_.n_resblocks = args_.n_resblocks / 2

                if args.model.lower() == 'rdn': 
                    args_.n_denseblocks = args_.n_denseblocks / 2 
                else: 
                    args_.n_layers = args_.n_layers / 2
                    
            args_.is_sub_mean = False

            module = import_module('model.{}'.format(args.model.lower()))
            branch = module.make_model(args_)

            if (not args.train_jointly) and (not (i == args.n_branches-1)): 
                for param in branch.parameters(): 
                    param.requires_grad = False

            self.branches.append(branch)
            self.add_module('recons_branch' if (i==0) else 'recons_branch_{}'.format(i),branch)

    def forward(self, x, y=None, train=False): 
        self.branch_outputs = [self.master_branch(x)]
        
        for i, branch in enumerate(self.branches): 
            
            if self.down_feats: 
                featmaps = self.master_branch.down_feats
            else: 
                if self.auto_feats: 
                    featmaps = self.master_branch.features[i]
                else: 
                    featmaps = self.master_branch.tail.modules().next()._modules['0'].outputs[i]

            self.branch_outputs.append(branch(featmaps))

        out = self.branch_outputs[-1] \
            if ((train==True) and (self.branch_label=='residual')) else \
                reduce(lambda x,y:x+y, self.branch_outputs)
        
        return out

    def load_master_state_dict(self, state_dict, strict=True):
        self.master_branch.load_state_dict(state_dict, strict)
