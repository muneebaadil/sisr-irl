import torch

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
        n_colors = args.n_colors 
        args.n_colors = 3 

        model_ref = model.Model(args, checkpoint)
        loader = data.Data(args, model_ref)

        args.n_colors = n_colors
        args.is_sub_mean = False
        args.pre_train = '.'
        
    model = model.Model(args, checkpoint)

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()