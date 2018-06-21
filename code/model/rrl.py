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

        self.lower_branches = []
        self.higher_branch = None 

        args.enable_rrl = False

        for branch_num in xrange(args.n_branches): 
            module = import_module('model.' + args.model.lower())
            branch = module.make_model(args)
            
            if not args.train_jointly: 
                for param in branch.parameters():
                    param.requires_grad = False

            self.lower_branches.append(branch)

            if args.half_feats: 
                args.n_feats = args.n_feats / (2**args.branch_num)

            if args.half_resblocks: 
                args.n_resblocks = args.n_resblocks / (2**args.branch_num)

            args.scale = [1]
            args.is_sub_mean = False

        args.pre_train = '.'
        args.n_channel_in = 64
        self.higher_branch = module.make_model(args)

        for i, branch in enumerate(self.lower_branches):
            self.add_module(str(i), branch)
        self.add_module(str(len(self.lower_branches)), self.higher_branch)

    def forward(self, x): 
        #IMPORTANT: ASSUMING ONLY ONE LOWER BRANCH FOR NOW..

        branch = self.lower_branches[0]
        out = branch(x)
        next_input = branch.featmaps

        out2 = self.higher_branch(next_input)
        out = out + out2

        return out

    def load_state_dict(self): 
        pass