from model import common 
import torch.nn as nn 
import pdb
import model
from importlib import import_module
from copy import deepcopy

def make_model(args, ckp, parents=False): 
    return RRL(args, ckp)

class RRL(nn.Module): 
    def __init__(self, args, ckp, conv=common.default_conv): 
        super(RRL, self).__init__()

        args.is_sub_mean = True 
        args_ = deepcopy(args)

        self.branches = []
        module = import_module('model.' + args.model.lower())
        self.master_branch = module.make_model(args)

        if not args.train_jointly: 
            for param in self.master_branch.parameters(): 
                param.requires_grad = False

        for i in xrange(args.n_branches):

            args_.n_channel_in = args.n_feats
            args_.scale = [args.scale[0] // 2] 
            if args.half_feats: 
                args_.n_feats = args.n_feats / 2 
            if args.half_resblocks: 
                args_.n_resblocks = args.n_resblocks / 2
            args_.is_sub_mean = False

            branch = module.make_model(args_)

            if (not args.train_jointly) and (not (i == args.n_branches-1)): 
                for param in branch.parameters(): 
                    param.requires_grad = False

            self.branches.append(branch)
            self.add_module('recons_branch' if (i==0) else 'recons_branch_{}'.format(i),branch)

    def forward(self, x): 
        # self.master_pred = self.master_branch(x)
        # self.featmaps = self.master_branch.tail.modules().next()._modules['0'].outputs[0]
        # self.refine_pred = self.recons_branch(self.featmaps)

        self.master_pred = self.master_branch(x)
        
        for i, branch in enumerate(self.branches): 
            featmaps = self.master_branch.tail.modules().next()._modules['0'].outputs[i]
            result = branch(featmaps)

            if not (i==len(self.branches)-1): 
                self.master_pred += result
            else: 
                self.refine_pred = result 

        return self.refine_pred

    # def load_state_dict(self, state_dict, strict=True): 
    #     self.master_branch.load_state_dict(state_dict, strict)

    # def load_state_dict2(self, state_dict, strict=True):
    #     self.recons_branch.load_state_dict(state_dict, strict)
