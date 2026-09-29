import logging
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils import MetricsTop, dict_to_str

logger = logging.getLogger('CyIN')

class CyIN():
    def __init__(self, args):
        self.args = args
        self.device = args.device

        if args.train_mode == 'regression':
            self.criterion = nn.L1Loss()
        elif args.train_mode == 'recognition':
            if args.dataset_name == 'meld':
                self.criterion = nn.CrossEntropyLoss()
            else:
                if 'class_weight' in args:
                    self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(args['class_weight']).to(args["device"]))
                else:
                    self.criterion = nn.CrossEntropyLoss()
        else:
            assert 0, "Undefined training mode (regression/recognition) !!!"

        self.metrics = MetricsTop(args.train_mode).getMetrics(args.dataset_name)

        from ..MissingProtocol import MissingProtocol
        self.miss_eval_protocol = MissingProtocol(args, device=self.device)

    def do_train(self, model, dataloader, return_epoch_results=False):
        all_params = list(model.Model.named_parameters())
        
        lm_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        lm_params_decay = [p for n, p in all_params if 'text_model' in n and not any(nd in n for nd in lm_no_decay)]
        lm_params_no_decay = [p for n, p in all_params if 'text_model' in n and any(nd in n for nd in lm_no_decay)]
        model_params_other = [p for n, p in all_params if 'text_model' not in n]

        optimizer_grouped_parameters = [
            {'params': lm_params_decay, 'weight_decay': self.args.weight_decay, 'lr': self.args.lr_lm},
            {'params': lm_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.lr_lm},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay, 'lr': self.args.lr}
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.learning_rate)
        
        # # Use transformer module
        # from transformers import get_linear_schedule_with_warmup
        # num_train_optimization_steps = (int(len(dataloader['train']) / self.args.batch_size) * self.args.max_epoch) # gradient_accumulation_step=1
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=num_train_optimization_steps * self.args.warmup_prop,
        #     num_training_steps=num_train_optimization_steps,
        # )

        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)

        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            task_loss_epoch, IB_loss_epoch, TRANSLATE_loss_epoch = 0.0, 0.0, 0.0
            left_epochs = self.args.grad_accum_epoch
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    # using accumulated gradients
                    if left_epochs == self.args.grad_accum_epoch:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    model.zero_grad()
                    text = batch_data['text']
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1) if self.args.train_mode == 'regression' else labels

                    # batch_size = labels.shape[0]
                    # train_mask = self.train_miss_aug(batch_size, 3, train_miss_prob=1.0)

                    # forward
                    self.stage_divide = self.args.stage_divide if 'stage_divide' in self.args else 1
                    if epochs <= self.args.max_epoch * self.stage_divide: 
                        generation_stage = False
                    else:
                        generation_stage = True

                    if generation_stage == False:
                        # stage 1 training for IB; stage 2 training for generation of Incomplete Mutlimodal Learning ##### default: 1/5
                        outputs = model(text, audio, vision, generation_stage=generation_stage) # , mask_matrix=train_mask
                    else: # stage 2 training for generation
                        outputs = model(text, audio, vision, generation_stage=generation_stage)

                    # compute loss
                    task_loss = self.criterion(outputs['M'], labels) # modality-shared
                    # if generation_stage == False:
                    for m in 'LAV':
                        task_loss += self.criterion(outputs[m], labels) # modality-specific label-level IB
                    
                    loss = task_loss + 1 / self.args.p_eta * outputs['IB_loss'] + self.args.w_imgaine * outputs['TRANSLATE_loss']
                    # backward
                    loss.backward()
                    if self.args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.args.grad_clip)
                    # store results
                    task_loss_epoch += task_loss.item()
                    IB_loss_epoch += outputs['IB_loss'].item()
                    TRANSLATE_loss_epoch += outputs['TRANSLATE_loss'].item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.grad_accum_epoch
                if not left_epochs:
                    # update
                    optimizer.step()
            task_loss_epoch = task_loss_epoch / len(dataloader['train'])
            IB_loss_epoch = IB_loss_epoch / len(dataloader['train'])
            TRANSLATE_loss_epoch = TRANSLATE_loss_epoch / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{best_epoch}/{epochs}] >> task_loss: {round(task_loss_epoch, 4)} IB_loss: {round(IB_loss_epoch, 4)} TRANSLATE_loss: {round(TRANSLATE_loss_epoch, 4)} {dict_to_str(train_results)}"
            )
            # # validation
            # val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            # cur_valid = val_results[self.args.KeyEval]
            # # scheduler.step(val_results['Loss'])
            # # save best model
            # isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # if isBetter:
            #     best_valid, best_epoch = cur_valid, epochs
            #     # save model
            #     torch.save(model.cpu().state_dict(), self.args.model_save_path)
            #     model.to(self.args.device)
            # # epoch results
            # if return_epoch_results:
            #     train_results["Loss"] = task_loss_epoch
            #     # # in debug
            #     # epoch_results['train'].append(train_results)
            #     # epoch_results['valid'].append(val_results)
            #     # test_results = self.do_test(model, dataloader['test'], mode="TEST")
            #     # epoch_results['test'].append(test_results)
            
            # save last model
            if epochs % 10 == 0 or epochs == self.args.max_epoch:
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            # early stop
            # if epochs - best_epoch >= self.args.early_stop:
            # max_epoch
            if epochs >= self.args.max_epoch:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text']
                    labels = batch_data['labels']['M'].to(self.device)
                    labels = labels.view(-1, 1) if self.args.train_mode == 'regression' else labels

                    batch_size = len(labels)
                    mask_matrix = self.miss_eval_protocol.get_mask_sample(3, batch_size) # [batchsize, 3] (L, A, V)

                    outputs = model(text, audio, vision, mask_matrix=mask_matrix, generation_stage=True) # , labels=labels, is_generation=True

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        # TODO: add features
                        # for item in features.keys():
                        #     features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        # test_preds_i = np.argmax(preds, axis=1)
                        sample_results.extend(preds.squeeze())
                    
                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            # for k in features.keys():
            #     features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
