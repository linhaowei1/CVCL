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
from networks.baseline.HAT import SGD_hat as SGD
from utils import utils
logger = logging.getLogger(__name__)

class Appr(object):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss()

    def train(self, model, train_loader, test_loaders, accelerator):
        
        no_decay = ["bias", "LayerNorm.weight"]
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.

        no_decay = ["bias", "LayerNorm.weight"]
        special_lr = []
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and p.requires_grad and not any(
                               nd in n for nd in special_lr)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and p.requires_grad and not any(
                               nd in n for nd in special_lr)],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate
            },
        ]

        if 'HAT' in self.args.baseline:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.num_train_epochs)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        if self.args.warmup_ratio:
            self.args.num_warmup_steps=self.get_warmup_steps(self.args.max_train_steps)

            print('self.args.num_warmup_steps: ',self.args.num_warmup_steps)

        model, train_loader = accelerator.prepare(model, train_loader)

        before_train.prepare(self.args, model, accelerator)

        # Train!
        total_batch_size = self.args.batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info("  Num examples = {}".format(len(train_loader) * self.args.batch_size))
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}, checkpoint Model = {self.args.model_name_or_path}")
        logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Learning Rate = {self.args.learning_rate}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        logger.info(f"  Seq ID = {self.args.idrandom}, Task id = {self.args.task}, Task Name = {self.args.task_name}, Num task = {self.args.ntasks}")

        if accelerator.is_main_process:
            tensorboard_file = os.path.join(self.args.output_dir, 'log')
            accelerator.print('tensorboard_file: ', tensorboard_file)
            if os.path.isdir(tensorboard_file):
                shutil.rmtree(tensorboard_file)
            writer = utils.init_writer(tensorboard_file)

        progress_bar = tqdm(range(self.args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        for epoch in range(starting_epoch, self.args.num_train_epochs):
            model.train()

            for step, batch in enumerate(train_loader):

                if 'HAT' in self.args.baseline:
                    s = (self.args.smax - 1 / self.args.smax) * step / len(
                        train_loader) + 1 / self.args.smax
                    outputs, masks = model(t=self.args.task, x=batch[0], s=s)
                    loss = self.cross_entropy(outputs, batch[1] - self.args.class_num * self.args.task)
                    loss += HAT.HAT_reg(self.args, masks)
                
                accelerator.backward(loss)

                if step % self.args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                    if 'HAT' in self.args.baseline:    
                        HAT.compensation(model, self.args, thres_cosh=self.args.thres_cosh, s=s)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clipgrad)
                        optimizer.step(hat=(self.args.task > 0))
                        HAT.compensation_clamp(model, thres_emb=6)
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_description(
                        'Train Iter (Epoch=%3d,loss=%5.3f)' % ((epoch, loss.item())))  # show the loss, mean while
            
            scheduler.step()
            
            if self.args.eval_during_training:
                results = self.eval(model, test_loaders, accelerator, eval_t=self.args.task)
                accelerator.print("*Epoch {}, til_acc = {}, cil_acc = {}, tp_acc = {}".format(\
                    epoch, results['til_accuracy'], results['cil_accuracy'], results['TP_accuracy']))
                utils.log(writer, 'til_accuracy', results['til_accuracy'], completed_steps)
                utils.log(writer, 'cil_accuracy', results['cil_accuracy'], completed_steps)
                utils.log(writer, 'TP_accuracy', results['TP_accuracy'], completed_steps)

            if completed_steps >= self.args.max_train_steps: break
        
        unwrapped_model = accelerator.unwrap_model(model)
        after_train.compute(self.args, unwrapped_model, accelerator)
        accelerator.wait_for_everyone()

        for eval_t in range(self.args.task + 1):

            accelerator.wait_for_everyone()
            results = self.eval(model, test_loaders, accelerator, eval_t)
            accelerator.print("*task {}, til_acc = {}, cil_acc = {}, tp_acc = {}".format(\
                    eval_t, results['til_accuracy'], results['cil_accuracy'], results['TP_accuracy']))

            if accelerator.is_main_process:
                utils.write_result(results, eval_t, self.args)


    def eval(self, model, test_loaders, accelerator, eval_t):
        
        model.eval()
        dataloader = accelerator.prepare(test_loaders[eval_t])
        label_list = []
        prediction_list = []
        taskscore_list = []
        total_num = 0
        for task_mask in range(self.args.task + 1):
            total_num = 0
            task_pred = []
            task_confidence = []
            task_label = []
            for _, batch in enumerate(dataloader):
                with torch.no_grad():
                    output, _ = model(task_mask, batch[0], s=self.args.smax)
                    score, prediction = torch.max(torch.softmax(output, dim=1), dim=1)

                    # for ddp
                    score = accelerator.gather(score)
                    predictions = accelerator.gather(prediction + task_mask * self.args.class_num)
                    references = accelerator.gather(batch[1])

                    total_num += batch[0].shape[0]
                    task_confidence += score.cpu().numpy().tolist()
                    task_label += references.cpu().numpy().tolist()
                    task_pred += predictions.cpu().numpy().tolist()
                
            label_list = task_label
            prediction_list.append(task_pred)
            taskscore_list.append(np.array(task_confidence))

        task_pred = np.argmax(np.stack(taskscore_list, axis=0), axis=0)
        cil_pred = [prediction_list[task_pred[i]][i] for i in range(total_num)]
        til_pred = [prediction_list[eval_t][i] for i in range(total_num)]

        cil_accuracy = sum(
            [1 if label_list[i] == cil_pred[i] else 0 for i in range(total_num)]
        ) / total_num
        til_accuracy = sum(
            [1 if label_list[i] == til_pred[i] else 0 for i in range(total_num)]
        ) / total_num
        TP_accuracy = sum(
            [1 if task_pred[i] == eval_t else 0 for i in range(total_num)]
        ) / total_num

        results = {
            'til_accuracy': round(til_accuracy, 4),
            'cil_accuracy': round(cil_accuracy, 4),
            'TP_accuracy': round(TP_accuracy, 4)
        }
        return results
            