import os
from networks.baseline import HAT
import numpy as np
import torch
from sklearn import metrics
import faiss
from torch.utils.tensorboard import SummaryWriter

def init_writer(name):
    writer = SummaryWriter(name)
    return writer

def log(writer, loss_name='training loss', scalar_value=None, global_step=None):
    # ...log the running loss
    writer.add_scalar(loss_name, scalar_value, global_step=global_step)

def write_result(results, eval_t, args):
    
    progressive_main_path = os.path.join(
        args.output_dir + '/../', 'progressive_main_' + str(args.seed)
    )
    progressive_til_path = os.path.join(
        args.output_dir + '/../', 'progressive_til_' + str(args.seed)
    )
    progressive_tp_path = os.path.join(
        args.output_dir + '/../', 'progressive_tp_' + str(args.seed)
    )

    if os.path.exists(progressive_main_path):
        eval_main = np.loadtxt(progressive_main_path)
    else:
        eval_main = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
    if os.path.exists(progressive_til_path):
        eval_til = np.loadtxt(progressive_til_path)
    else:
        eval_til = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
    if os.path.exists(progressive_tp_path):
        eval_tp = np.loadtxt(progressive_tp_path)
    else:
        eval_tp = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
    
    eval_main[args.task][eval_t] = results['cil_accuracy']
    eval_til[args.task][eval_t] = results['til_accuracy']
    eval_tp[args.task][eval_t] = results['TP_accuracy']

    np.savetxt(progressive_main_path, eval_main, '%.4f', delimiter='\t')
    np.savetxt(progressive_til_path, eval_til, '%.4f', delimiter='\t')
    np.savetxt(progressive_tp_path, eval_tp, '%.4f', delimiter='\t')
    
    if args.task == args.ntasks - 1:
        final_main = os.path.join(args.output_dir + '/../', 'final_main_' + str(args.seed))
        forward_main = os.path.join(args.output_dir + '/../', 'forward_main_' + str(args.seed))
        
        final_til = os.path.join(args.output_dir + '/../', 'final_til_' + str(args.seed))
        forward_til = os.path.join(args.output_dir + '/../', 'forward_til_' + str(args.seed))

        final_tp = os.path.join(args.output_dir + '/../', 'final_tp_' + str(args.seed))
        forward_tp = os.path.join(args.output_dir + '/../', 'forward_tp_' + str(args.seed))

        with open(final_main, 'w') as final_main_file, open(forward_main, 'w') as forward_main_file:
            for j in range(eval_main.shape[1]):
                final_main_file.writelines(str(eval_main[-1][j]) + '\n')
                forward_main_file.writelines(str(eval_main[j][j]) + '\n')
        
        with open(final_til, 'w') as final_til_file, open(forward_til, 'w') as forward_til_file:
            for j in range(eval_til.shape[1]):
                final_til_file.writelines(str(eval_til[-1][j]) + '\n')
                forward_til_file.writelines(str(eval_til[j][j]) + '\n')
        
        with open(final_tp, 'w') as final_tp_file, open(forward_tp, 'w') as forward_tp_file:
            for j in range(eval_tp.shape[1]):
                final_tp_file.writelines(str(eval_tp[-1][j]) + '\n')
                forward_tp_file.writelines(str(eval_tp[j][j]) + '\n')
                    
def prepare_sequence_eval(args):
    with open(os.path.join('./sequence', args.sequence_file), 'r') as f:
        data = f.readlines()[args.idrandom]
        data = data.split()

    args.all_tasks = data
    args.ntasks = len(data)
    ckpt = args.base_dir + '/seq' + str(args.idrandom) + "/seed" + str(args.seed) + '/' + str(args.baseline) + '/' + str(data[args.ntasks-1]) + '/model'

    args.prev_output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(args.baseline) + '/' + str(data[args.ntasks-1])
    args.model_name_or_path = ckpt
    args.output_dir = None

    print('output_dir: ', args.output_dir)
    print('prev_output: ', args.prev_output)
    print('args.model_name_or_path: ', args.model_name_or_path)

    return args

def prepare_sequence_train(args):
    with open(os.path.join('./sequence', args.sequence_file), 'r') as f:
        data = f.readlines()[args.idrandom]
        data = data.split()

    args.task_name = data[args.task]
    args.all_tasks = data
    args.ntasks = len(data)
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

def auroc(predictions, references):
    fpr, tpr, _ = metrics.roc_curve(references, predictions, pos_label=1)
    return metrics.auc(fpr, tpr)

def KNN(args, results, train_hidden):
    
    metric = {}

    for train_t in range(args.ntasks):
        
        index = faiss.IndexFlatL2(len(train_hidden[0][0][0]))
        index.add(np.array(train_hidden[train_t][train_t]).astype(np.float32))   # the train_t taskmask on the train_t th dataset
        index_out = faiss.IndexFlatL2(len(train_hidden[0][0][0]))
        for t in range(args.ntasks):
            if t == train_t: continue
            index_out.add(np.array(train_hidden[t][train_t]).astype(np.float32))
        
        print(len(train_hidden[train_t][train_t]))

        distance = {}
        metric[train_t] = {}
        
        auc_reference = []
        auc_prediction = []
        softmax_prediction = []

        for eval_t in range(args.ntasks):
            # the train_t th task mask on eval_t dataset
            D = index.search(np.array(results[eval_t]['hidden'][train_t]).astype(np.float32), args.K)
            D_out = index_out.search(np.array(results[eval_t]['hidden'][train_t]).astype(np.float32), args.K * (args.ntasks-1))

            distance[eval_t] = - (D[0][:, -1] / D_out[0][:, -1])

            auc_reference += [int(eval_t == train_t)] * len(distance[eval_t])
            auc_prediction += distance[eval_t].tolist()
            softmax_prediction += results[eval_t]['softmax_prob'][train_t]
        
        metric[train_t]['auc'] = auroc(auc_prediction, auc_reference)
        metric[train_t]['softmax_auc'] = auroc(softmax_prediction, auc_reference)
    
    print(metric)

