import torch
import pdb

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    if args.data_train.lower()=='rrl' or args.data_test.lower()=='rrl':     
        args.is_sub_mean = True 
        n_channel_in = args.n_channel_in 
        args.n_channel_in = 3
        cpu = args.cpu 
        args.cpu = True

        model_ref = model.Model(args, checkpoint)
        loader = data.Data(args, model_ref)

        args.n_channel_in = n_channel_in
        args.is_sub_mean = False
        args.pre_train = '.'
        args.scale = [args.scale[0] / (2**args.branch_num)]
        args.cpu = cpu

        if args.half_feats: 
            args.num_feats = args.num_feats / (2**args.branch_num)

        if args.half_resblocks: 
            args.num_resblocks = args.num_resblocks / (2**args.branch_num)

        for p in model_ref.parameters(): 
            p.requires_grad = False

    else: 
        args.is_sub_mean = True 
        loader = data.Data(args, None)
        
    model = model.Model(args, checkpoint)

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()