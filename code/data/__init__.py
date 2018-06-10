from importlib import import_module

from dataloader import MSDataLoader
from torch.utils.data.dataloader import default_collate

import pdb

class Data:
    def __init__(self, args, model):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:

            if args.data_train.lower() != 'rrl':
                module_train = import_module('data.' + args.data_train.lower())
                trainset = getattr(module_train, args.data_train)(args)
            else: 
                module_train = import_module('data.' + args.rrl_data.lower())
                trainclass = getattr(module_train, args.rrl_data)
                
                module_train = import_module('data.rrl')
                trainset = getattr(module_train, 'RRL')(trainclass, args, model)

            self.loader_train = MSDataLoader(
                    args,
                    trainset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    **kwargs
                )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            if not args.benchmark_noise:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )

        else:
            if args.data_test.lower() != 'rrl': 
                module_test = import_module('data.' +  args.data_test.lower())
                testset = getattr(module_test, args.data_test)(args, train=False)
            else: 
                module_test = import_module('data.' + args.rrl_data.lower())
                testclass = getattr(module_test, args.rrl_data)

                module_test = import_module('data.rrl')
                testset = getattr(module_test, 'RRL')(testclass, args, model, False)

        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )
