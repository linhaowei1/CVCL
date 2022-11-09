from modulefinder import IMPORT_NAME
from approaches import before_train, after_train
from tqdm.auto import tqdm
import torch.nn as nn
import os
import shutil
import torch
from networks.baseline import HAT
import numpy as np
import logging
import math
from transformers import get_scheduler
from networks.baseline.HAT import Adam
from utils import utils
logger = logging.getLogger(__name__)

class Appr(object):
    
    def __init__(self, args):
        super().__init__()
        self.args = args

    def eval(self, model, train_loaders, test_loaders, accelerator):
        
        model = accelerator.prepare(model)
        
        before_train.prepare(self.args, model, accelerator)

        results = {}
        train_hidden = {}
        
        model.eval()

        for eval_t in range(self.args.task + 1):

            results[eval_t] = {
                'predictions': [],      # [N x data], prediction of N task mask
                'references': [],       # [data]
                'hidden': [],           # [N x data]
                'estimate_prob': [],    # [N x data]
                'softmax_prob': [],     # [N x data]
                'total_num': 0
            }
            train_hidden[eval_t] = []

            test_loader, train_loader = accelerator.prepare(test_loaders[eval_t], train_loaders[eval_t])

            for task_mask in range(self.args.task + 1):
                
                train_hidden_list = []
                hidden_list = []
                prediction_list = []
                softmax_list = []

                for _, batch in enumerate(test_loader):
                    with torch.no_grad():
                        output, _, hidden = model(task_mask, batch[0], s=self.args.smax, output_hidden=True)
                        score, prediction = torch.max(torch.softmax(output, dim=1), dim=1)

                        hidden_list += nn.functional.normalize(hidden).cpu().numpy().tolist()
                        prediction_list += (prediction + self.args.class_num * eval_t).cpu().numpy().tolist()
                        softmax_list += score.cpu().numpy().tolist()

                        if task_mask == 0:
                            results[eval_t]['total_num'] += batch[0].shape[0]
                            results[eval_t]['references'] += batch[1].cpu().numpy().tolist()
                
                results[eval_t]['hidden'].append(hidden_list)
                results[eval_t]['predictions'].append(prediction_list)
                results[eval_t]['softmax_prob'].append(softmax_list)
                
                for _, batch in enumerate(train_loader):
                    with torch.no_grad():
                        output, _, hidden = model(task_mask, batch[0], s=self.args.smax, output_hidden=True)
                        train_hidden_list += nn.functional.normalize(hidden).cpu().numpy().tolist()
                
                train_hidden[eval_t].append(train_hidden_list)
        
        utils.KNN(self.args, results, train_hidden)
            