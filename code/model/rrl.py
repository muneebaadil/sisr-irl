from model import common 
import torch.nn as nn 
import model

def make_model(args, ckp, parents=False): 
    return RRL(args, ckp)

class RRL(nn.Module): 
    def __init__(self, args, ckp, conv=common.default_conv): 
        super(RRL, self).__init__()

        self.lower_branches = []
        self.higher_branch = None 

        args.enable_rrl = False

        for branch_num in xrange(args.n_branches): 
            branch = model.Model(args, ckp)
            
            if not args.train_jointly: 
                for param in branch.parameters():
                    param.requires_grad = False

            self.lower_branches.append(branch)

            if args.half_feats: 
                args.n_feats = args.n_feats / (2**args.branch_num)

            if args.half_resblocks: 
                args.n_resblocks = args.n_resblocks / (2**args.branch_num)

        self.higher_branch = model.Model(args, ckp)


    def to(self, device): 
        for branch in self.lower_branches: 
            branch.to(device)
        
        self.higher_branch.to(device)

    def forward(self, x): 
        pass 

    def load_state_dict(self): 
        pass