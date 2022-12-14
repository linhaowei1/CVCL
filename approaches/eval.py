from modulefinder import IMPORT_NAME
from approaches import before_train, after_train
from tqdm.auto import tqdm
import torch.nn as nn
import os
import shutil
import torch
import json
import numpy as np
import logging
import math
from transformers import get_scheduler
from utils import utils
logger = logging.getLogger(__name__)

class Appr(object):
    
    def __init__(self, args):
        super().__init__()
        self.args = args

    def eval(self, model, train_loaders, test_loaders, accelerator):
        
        model = accelerator.prepare(model)
        
        #before_train.prepare(self.args, model, accelerator)
        if os.path.exists(os.path.join(self.args.output_dir, 'train_hidden')):
            with open(os.path.join(self.args.output_dir, 'results'), 'r') as f:
                results = json.load(f)
                results = {int(k): v for (k,v) in results.items()}
            with open(os.path.join(self.args.output_dir, 'train_hidden'), 'r') as f:
                train_hidden = json.load(f)
                train_hidden = {int(k): v for (k,v) in train_hidden.items()}
            with open(os.path.join(self.args.output_dir, 'train_labels'), 'r') as f:
                train_labels = json.load(f)
                train_labels = {int(k): v for (k,v) in train_labels.items()}
            with open(os.path.join(self.args.output_dir, 'train_logits'), 'r') as f:
                train_logits = json.load(f)
                train_logits = {int(k): v for (k,v) in train_logits.items()}
        else:
            results = {}
            train_hidden = {}
            train_labels = {}
            train_logits = {}
            model.eval()

            for eval_t in tqdm(range(self.args.task + 1)):

                results[eval_t] = {
                    'predictions': [],      # [N x data], prediction of N task mask
                    'references': [],       # [data]
                    'hidden': [],           # [N x data]
                    'logits': [],    # [N x data]
                    'softmax_prob': [],     # [N x data]
                    'total_num': 0
                }
                train_hidden[eval_t] = []
                train_labels[eval_t] = []
                train_logits[eval_t] = []
                test_loader, train_loader = accelerator.prepare(test_loaders[eval_t], train_loaders[eval_t])

                for task_mask in range(self.args.task + 1):
                    
                    train_hidden_list = []
                    hidden_list = []
                    prediction_list = []
                    logits_list = []
                    softmax_list = []
                    train_logits_list = []

                    for _, batch in enumerate(test_loader):
                        with torch.no_grad():
                            features, _ = model.forward_features(task_mask, batch[0], s=self.args.smax)
                            output = model.forward_classifier(task_mask, features)
                            score, prediction = torch.max(torch.softmax(output, dim=1), dim=1)

                            hidden_list += (features).cpu().numpy().tolist()
                            prediction_list += (prediction + self.args.class_num * task_mask).cpu().numpy().tolist()
                            softmax_list += score.cpu().numpy().tolist()
                            logits_list += output.cpu().numpy().tolist()

                            if task_mask == 0:
                                results[eval_t]['total_num'] += batch[0].shape[0]
                                results[eval_t]['references'] += batch[1].cpu().numpy().tolist()
                    
                    results[eval_t]['hidden'].append(hidden_list)
                    results[eval_t]['predictions'].append(prediction_list)
                    results[eval_t]['softmax_prob'].append(softmax_list)
                    results[eval_t]['logits'].append(logits_list)

                    
                    for _, batch in enumerate(train_loader):
                        with torch.no_grad():
                            features, _ = model.forward_features(task_mask, batch[0], s=self.args.smax)
                            output = model.forward_classifier(task_mask, features)
                            train_logits_list += output.cpu().numpy().tolist()
                            train_hidden_list += (features).cpu().numpy().tolist()
                            if task_mask == 0:
                                train_labels[eval_t] += (batch[1] - self.args.class_num * eval_t).cpu().numpy().tolist()
                    
                    train_hidden[eval_t].append(train_hidden_list)
                    train_logits[eval_t].append(train_logits_list)

            with open(os.path.join(self.args.output_dir, 'results'), 'w') as f:
                json.dump(results, f)
            with open(os.path.join(self.args.output_dir, 'train_hidden'), 'w') as f:
                json.dump(train_hidden, f)
            with open(os.path.join(self.args.output_dir, 'train_labels'), 'w') as f:
                json.dump(train_labels, f)
            with open(os.path.join(self.args.output_dir, 'train_logits'), 'w') as f:
                json.dump(train_logits, f)

        #utils.KNN(self.args, results, train_hidden)
        utils.pout(self.args, results, train_hidden, train_labels, train_logits)
