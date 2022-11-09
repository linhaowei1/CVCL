import os
import torch
def prepare(args, model, accelerator):

    if 'HAT' in args.baseline:
        args.mask_pre = None
        args.mask_back = None
        if 'C10_5T' in args.baseline:
            args.reg_lambda = 1.0 if args.task == 0 else 0.75
        elif 'C100_10T' in args.baseline:
            args.reg_lambda = 1.5 if args.task == 0 else 1.0
        elif 'C100_20T' in args.baseline:
            args.reg_lambda = 3.5 if args.task == 0 else 2.5
        if args.task > 0:
            print('load mask matrix ....')
            args.mask_pre = torch.load(os.path.join(args.prev_output, 'mask_pre'), map_location='cpu')
            args.mask_back = torch.load(os.path.join(args.prev_output, 'mask_back'), map_location='cpu')

            for k, v in args.mask_pre.items():
                args.mask_pre[k] = args.mask_pre[k].to(accelerator.device)

            for k, v in args.mask_back.items():
                args.mask_back[k] = args.mask_back[k].to(accelerator.device)
        
            for n, p in model.named_parameters():
                p.grad = None
                if n in args.mask_back.keys():
                    p.hat = args.mask_back[n]
                else:
                    p.hat = None