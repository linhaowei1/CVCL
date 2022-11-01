import os
from networks.baseline import HAT
import torch
def compute(args, model, accelerator):
    
    if 'HAT' in args.baseline:
        args.mask_pre = HAT.cum_mask(smax=args.smax, t=args.task, model=model, mask_pre=args.mask_pre, accelerator=accelerator)
        args.mask_back = HAT.freeze_mask(args.smax, args.task, model, args.mask_pre)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save(args.mask_pre, os.path.join(args.output_dir, 'mask_pre'))
            torch.save(args.mask_back, os.path.join(args.output_dir, 'mask_back'))
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model'))