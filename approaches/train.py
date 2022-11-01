from modulefinder import IMPORT_NAME
from approaches import before_train, after_train
from tqdm.auto import tqdm
import torch.nn as nn
import torch
from networks.baseline import HAT
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
            optimizer = Adam(optimizer_grouped_parameters)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        if self.args.warmup_ratio:
            self.args.num_warmup_steps=self.get_warmup_steps(self.args.max_train_steps)

            print('self.args.num_warmup_steps: ',self.args.num_warmup_steps)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )
        model, train_loader, lr_scheduler = accelerator.prepare(model, train_loader, lr_scheduler)

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
                        HAT.compensation(model, self.args, thres_cosh=self.args.thres_cosh, s=1)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clipgrad)
                        optimizer.step(hat=(self.args.task > 0))
                        HAT.compensation_clamp(model, thres_emb=6)
                    else:
                        optimizer.step()

                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_description(
                        'Train Iter (Epoch=%3d,loss=%5.3f)' % ((epoch, loss.item())))  # show the loss, mean while
            
            if self.args.eval_during_training:
                results = self.eval(model, test_loaders, accelerator, eval_t=self.args.task)
                accelerator.print("*Epoch {}, test_acc = {}, test_loss = {}".format(epoch, results['accuracy'], results['loss']))

            if completed_steps >= self.args.max_train_steps: break
        
        unwrapped_model = accelerator.unwrap_model(model)
        after_train.compute(self.args, unwrapped_model, accelerator)
        accelerator.wait_for_everyone()

        for eval_t in range(self.args.task + 1):

            accelerator.wait_for_everyone()
            results = self.eval(model, test_loaders, accelerator, eval_t)
            accelerator.print("task: {}, test_acc = {}, test_loss = {}".format(eval_t, results['accuracy'], results['loss']))

            if accelerator.is_main_process:
                utils.write_result(results, eval_t, self.args)


    def eval(self, model, test_loaders, accelerator, eval_t):
        
        model.eval()
        dataloader = accelerator.prepare(test_loaders[eval_t])
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        label_list = []
        prediction_list = []
        total_loss = 0.0
        total_num = 0
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                output, _ = model(eval_t, batch[0], s=self.args.smax)
                loss = self.cross_entropy(output, batch[1] - eval_t * self.args.class_num)
                prediction = torch.argmax(output, dim=1)
                predictions = accelerator.gather(prediction)
                references = accelerator.gather(batch[1] - eval_t * self.args.class_num)

                total_loss += loss.data.cpu().numpy().item() * batch[0].shape[0]
                total_num += batch[0].shape[0]
                label_list += references.cpu().numpy().tolist()
                prediction_list += predictions.cpu().numpy().tolist()

                progress_bar.update(1)

        accuracy = sum([1 if label_list[i] == prediction_list[i] else 0 for i in range(len(label_list))]) / total_num
        loss = total_loss / total_num

        results = {
            'accuracy': round(accuracy, 4),
            'loss': round(loss, 4)
        }
        return results
            