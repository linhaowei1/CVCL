import os
from networks.vit_hat import deit_small_patch16_224
from networks.clip_hat import build_model
import numpy as np
import torch
import sklearn.covariance
from sklearn import metrics
import json
import faiss
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
def init_writer(name):
    writer = SummaryWriter(name)
    return writer

def log(writer, loss_name='training loss', scalar_value=None, global_step=None):
    # ...log the running loss
    writer.add_scalar(loss_name, scalar_value, global_step=global_step)

def normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v, axis=-1).reshape(-1,1)
    return v / (norm + 1e-9)

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

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
    args.output_dir = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(args.baseline)

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

    if 'contras' in args.baseline and not args.contras:
        args.prev_output = args.base_dir + '/seq' + str(args.idrandom) + "/seed" + str(args.seed) + '/' + str(args.baseline) + '/' + str(data[args.task])
        args.model_name_or_path = args.base_dir + '/seq' + str(args.idrandom) + "/seed" + str(args.seed) + '/' + str(args.baseline) + '/' + str(data[args.task]) + '/model'
    elif args.task > 0:
        args.prev_output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(args.baseline) + '/' + str(data[args.task-1])
        args.model_name_or_path = ckpt
    else:
        args.prev_output = None
        args.model_name_or_path = None

    print('output_dir: ', args.output_dir)
    print('prev_output: ', args.prev_output)
    print('args.model_name_or_path: ', args.model_name_or_path)

    return args

def load_deit_pretrain(args, target_model):
    """
        target_model: the model we want to replace the parameters (most likely un-trained)
    """
    if os.path.isfile('/home/haowei/haowei/MORE/deit_pretrained/best_checkpoint.pth'):
        checkpoint = torch.load('/home/haowei/haowei/MORE/deit_pretrained/best_checkpoint.pth', map_location='cpu')
    else:
        raise NotImplementedError("Cannot find pre-trained model")
    target = target_model.state_dict()
    pretrain = checkpoint['model']
    transfer = {k: v for k, v in pretrain.items() if k in target and 'head' not in k}
    target.update(transfer)
    target_model.load_state_dict(target)
    
def lookfor_model(args):
    
    if 'more' in args.baseline:
        if 'clip' in args.baseline:
            state_dict = torch.jit.load('/data1/haowei/haowei/cl/ViT-B-32.pt', map_location='cpu').state_dict()
            model = build_model(state_dict, latent=args.latent, num_classes=args.class_num)
        else:
            model = deit_small_patch16_224(pretrained=True, num_classes=args.class_num, latent=args.latent, args=args)
            load_deit_pretrain(args, model)
        for _ in range(args.task):
            model.append_embeddings()
        if args.task > 0:
            if not args.training:
                model.append_embeddings()
            model.load_state_dict(torch.load(os.path.join(args.model_name_or_path), map_location='cpu'))
        if args.training:
            model.append_embeddings()
        
    else:
        raise NotImplementedError
    return model

def auroc(predictions, references):
    fpr, tpr, _ = metrics.roc_curve(references, predictions, pos_label=1)
    return metrics.auc(fpr, tpr)

def acc(predictions, references):
    acc = metrics.accuracy_score(references, predictions)
    return acc

@torch.no_grad()
def Mahainit(args, train_hidden, train_labels):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    feat_mean_list = {} # ntasks x ntasks x num_classes
    precision_list = {}
    feat_list = {}
    for train_t in tqdm(range(args.ntasks)):
        # prepare feat_mean_list, precision_lst
        feat_list[train_t] = {}
        feat_mean_list[train_t] = {}
        precision_list[train_t] = {}
        for task_t in range(args.ntasks):
            feat_list[train_t][task_t] = {}
            for feature, label in zip(train_hidden[train_t][task_t], train_labels[train_t]):
                feature = np.array(feature).reshape([-1, len(feature)])
                if label not in feat_list[train_t][task_t].keys():
                    feat_list[train_t][task_t][label] = feature
                else:
                    feat_list[train_t][task_t][label] = np.concatenate(
                        (feat_list[train_t][task_t][label], feature), axis=0)

            feat_mean_list[train_t][task_t] = [np.mean(feat_list[train_t][task_t][i], axis=0).tolist() for i in range(args.class_num)]
            precision_list[train_t][task_t] = []
            for k in range(args.class_num):
                X = feat_list[train_t][task_t][k] - feat_mean_list[train_t][task_t][k]
                # find inverse
                group_lasso.fit(X)
                precision = group_lasso.precision_
                precision_list[train_t][task_t].append(precision.tolist())

    return feat_mean_list, precision_list


def calibration(logits, label):
    # logits: N x task_num x class_num
    logits = torch.from_numpy(logits).cuda()
    label = torch.from_numpy(label).cuda()

    from torch.optim import Adam
    import torch.nn as nn
    w, b = torch.randn(1, logits.shape[1], 1).cuda(), torch.randn(1, logits.shape[1], 1).cuda()
    w.requires_grad = True
    b.requires_grad = True
    optimizer = Adam([w, b], lr=0.1)

    for _ in tqdm(range(500)):
        loss = nn.CrossEntropyLoss()((logits * torch.abs(w.expand_as(logits)) + b.expand_as(logits)).reshape(logits.shape[0], -1), label)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return w.cpu().squeeze(), b.cpu().squeeze()

def pout(args, results, train_hidden, train_labels, train_logits):
    print("start mahainit...")
    if os.path.exists(os.path.join(args.output_dir, 'feat_mean_list')):
        with open(os.path.join(args.output_dir, 'feat_mean_list'), 'r') as f:
            feat_mean_list = json.load(f)
        with open(os.path.join(args.output_dir, 'precision_list'), 'r') as f:
            precision_list = json.load(f)
    else:
        feat_mean_list, precision_list = Mahainit(args, train_hidden, train_labels)
        with open(os.path.join(args.output_dir, 'feat_mean_list'), 'w') as f:
            json.dump(feat_mean_list, f)
        with open(os.path.join(args.output_dir, 'precision_list'), 'w') as f:
            json.dump(precision_list, f)
    print("finish mahainit!!")

    index_out = {}

    for train_t in range(args.ntasks):
        
        index_out[train_t] = faiss.IndexFlatL2(len(train_hidden[0][0][0]))
        for t in range(args.ntasks):
            if t == train_t: continue
            index_out[train_t].add(normalize(train_hidden[t][train_t][:args.replay_buffer_size // args.class_num]).astype(np.float32))        
    
    # ## calibration 
    # calibration_logits = []
    # calibration_label = []

    # for train_t in range(args.ntasks):

    #     logits = np.transpose(train_logits[train_t], (1, 0, 2))     # [data, task_num, class_num]
    #     softmax = torch.softmax(torch.from_numpy(logits[:40] / 2), dim=-1)
    #     calibration_logits.append(logits[:40]) # [data, task_num, class_num]
    #     calibration_label.append(train_labels[train_t][:40]) # [data, class_num]

    # w, b = calibration(np.concatenate(calibration_logits, axis=0), np.concatenate(calibration_label, axis=0))

    metric = {}

    for eval_t in range(args.ntasks):
        
        # [task_num, test_sample_num, class_num]
        metric[eval_t] = {}

        logits = np.transpose(results[eval_t]['logits'], (1, 0, 2))
        # logits = logits * abs(w[eval_t].item()) + b[eval_t].item()
        softmax = torch.softmax(torch.from_numpy(logits / 2), dim=-1)

        for task_mask in range(args.ntasks):

            test_sample = torch.Tensor(results[eval_t]['hidden'][task_mask])
            D_out = index_out[task_mask].search(
                normalize(test_sample).astype(np.float32), args.K)
            
            for class_idx in range(args.class_num):
                zero_f = test_sample - torch.Tensor(feat_mean_list[str(task_mask)][str(task_mask)][class_idx])
                term_gau =  1 / torch.mm(torch.mm(zero_f, torch.Tensor(precision_list[str(task_mask)][str(task_mask)][class_idx])),
                                        zero_f.t()).diag()
                if class_idx == 0:
                    noise_gaussian_score = term_gau.view(-1, 1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)
                
            score_in, _ = torch.max(noise_gaussian_score, dim=1)
            score_in = score_in * torch.tensor(D_out[0][:, -1]) 
            softmax[: ,task_mask] *= score_in.unsqueeze(-1) 
        
        prediction = np.argmax(softmax.reshape(-1, args.ntasks * args.class_num), axis=1)
        metric[eval_t]['acc'] = acc(prediction, results[eval_t]['references'])
    
    print(metric)
    print("acc:", np.average([metric[i]['acc'] for i in range(args.ntasks)]))



def KNN(args, results, train_hidden):
    
    metric = {}
    index_in = {}
    index_out = {}

    for train_t in range(args.ntasks):
        
        index_in[train_t] = faiss.IndexFlatL2(len(train_hidden[0][0][0]))
        index_in[train_t].add(normalize(train_hidden[train_t][train_t][:40]).astype(np.float32))   # the train_t taskmask on the train_t th dataset
        index_out[train_t] = faiss.IndexFlatL2(len(train_hidden[0][0][0]))
        for t in range(args.ntasks):
            if t == train_t: continue
            index_out[train_t].add(normalize(train_hidden[t][train_t][:40]).astype(np.float32))
        
    calibration_logits = []
    calibration_label = []

    for train_t in range(args.ntasks):

        tp_logits = []
        for task_mask in range(args.ntasks):
            D_in = index_in[task_mask].search(
                normalize(train_hidden[train_t][task_mask][:40]).astype(np.float32), args.K)
            D_out = index_out[task_mask].search(
                normalize(train_hidden[train_t][task_mask][:40]).astype(np.float32), args.K)
            tp_logits.append(D_out[0][:, -1] / D_in[0][:, -1])

        calibration_logits.append(np.stack(tp_logits, axis=0))
        calibration_label += [train_t] * len(tp_logits[0])

    w, b = calibration(np.transpose(np.concatenate(calibration_logits, axis=1)), np.array(calibration_label))


    for eval_t in range(args.ntasks):
        
        tp_logits = []

        for task_mask in range(args.ntasks):
            D_in = index_in[task_mask].search(
                normalize(results[eval_t]['hidden'][task_mask]).astype(np.float32), args.K)
            D_out = index_out[task_mask].search(
                normalize(results[eval_t]['hidden'][task_mask]).astype(np.float32), args.K)
            tp_logits.append(D_out[0][:, -1] / D_in[0][:, -1])
        
        with torch.no_grad():
            tp_logits = (torch.from_numpy(np.transpose(np.stack(tp_logits, axis=0))) * w + b).detach().numpy()

        knn_task_pred = np.argmax(tp_logits, axis=1)
        pred_knn = [results[eval_t]['predictions'][pp][i] for i, pp in enumerate(knn_task_pred)]
        softmax_task_pred = np.argmax(np.stack(results[eval_t]['softmax_prob'], axis=0), axis=0)
        pred_softmax = [results[eval_t]['predictions'][pp][i] for i, pp in enumerate(softmax_task_pred)]

        metric[eval_t] = {}
        metric[eval_t]['acc_knn'] = acc(pred_knn, results[eval_t]['references'])
        metric[eval_t]['acc_softmax'] = acc(pred_softmax, results[eval_t]['references'])
    
    print("knn_acc:", np.average([metric[i]['acc_knn'] for i in range(args.ntasks)]))
    print("softmax_acc:", np.average([metric[i]['acc_softmax'] for i in range(args.ntasks)]))
    print(metric)



