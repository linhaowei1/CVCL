import os
from networks.baseline import HAT
import numpy as np
import torch

def write_result(results, eval_t, args):
    progressive_main_path = os.path.join(
        args.output_dir + '/../', 'progressive_main_' + str(args.seed)
    )
    if os.path.exists(progressive_main_path):
        eval_main = np.loadtxt(progressive_main_path)
    else:
        eval_main = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
    
    eval_main[args.task][eval_t] = results['accuracy']

    np.savetxt(progressive_main_path, eval_main, '%.4f', delimiter='\t')
    
    if args.task == args.ntasks - 1:
        final_main = os.path.join(args.output_dir, 'final_main_' + str(args.seed))
        forward_main = os.path.join(args.output_dir, 'forward_main_' + str(args.seed))

        with open(final_main, 'w') as final_main_file, open(forward_main, 'w') as forward_main_file:
            for j in range(eval_main.shape[1]):
                final_main_file.writelines(str(eval_main[-1][j]) + '\n')
                forward_main_file.writelines(str(eval_main[j][j]) + '\n')
                    


def prepare_sequence(args):
    with open(os.path.join('./sequence', args.sequence_file), 'r') as f:
        data = f.readlines()[args.idrandom]
        data = data.split()

    args.task_name = data[args.task]
    args.all_tasks = data

    args.output_dir = args.base_dir + '/seq' + str(args.idrandom) + "/seed" + str(args.seed) + '/' + str(args.baseline) + '/' + str(data[args.task])
    ckpt = args.base_dir + '/seq' + str(args.idrandom) + "/seed" + str(args.seed) + '/' + str(args.baseline) + '/' + str(data[args.task-1]) + '/model'

    if args.task > 0:
        args.prev_output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(args.baseline) + '/' + str(data[args.task-1])
        args.model_name_or_path = ckpt
    else:
        args.prev_output = None
        args.model_name_or_path = None

    print('output_dir: ', args.output_dir)
    print('prev_output: ', args.prev_output)
    print('args.model_name_or_path: ', args.model_name_or_path)

    return args

def lookfor_model(args):
    
    if 'HAT' in args.baseline:
        model = HAT.ResNet18(num_classes=args.class_num, total_classes=args.total_class)
        if args.model_name_or_path is not None:
            model.load_state_dict(torch.load(args.model_name_or_path, map_location='cpu'))
    else:
        raise NotImplementedError
    return model