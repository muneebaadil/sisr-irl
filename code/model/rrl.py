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

        module = import_module('model.edsr')
        self.master_branch = module.make_model(args)
        
        args.n_channel_in = 64
        args.scale = [1]
        self.recons_branch = module.make_model(args)
        # args.enable_rrl = False

        # for branch_num in xrange(args.n_branches): 
        #     module = import_module('model.' + args.model.lower())
        #     branch = module.make_model(args)
            
        #     if not args.train_jointly: 
        #         for param in branch.parameters():
        #             param.requires_grad = False

        #     self.lower_branches.append(branch)

        #     if args.half_feats: 
        #         args.n_feats = args.n_feats / (2**branch_num)

        #     if args.half_resblocks: 
        #         args.n_resblocks = args.n_resblocks / (2**branch_num)

        #     args.scale = [1]
        #     args.is_sub_mean = False

        # pre_train = args.pre_train 
        # args.pre_train = '.'
        # args.n_channel_in = 64
        # self.higher_branch = module.make_model(args)

        # args.pre_train = pre_train

    def forward(self, x): 
        #IMPORTANT: ASSUMING ONLY ONE LOWER BRANCH FOR NOW..

        # branch = self.lower_branches[0]
        # self.lower_out = branch(x)
        # print 'lower branch done'
        # next_input = branch.featmaps

        # self.higher_out = self.higher_branch(next_input)
        # print 'higher branch done'
        # self.final_out = self.lower_out + self.higher_out

        y1 = self.master_branch(x)
        y2 = self.recons_branch(self.master_branch.featmaps)

        out = y1 + y2

        return out

    def load_state_dict(self, state_dict, strict=True): 
        self.master_branch.load_state_dict(state_dict, strict)