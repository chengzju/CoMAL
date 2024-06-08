import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from losses import *
# from metric import base1_metric, label2list, pair_metric_ps
from apex import amp
import torch
from collections import OrderedDict
from future.utils import iteritems
from utils import *
from metircs import *
from config import *
import queue
import time


def convert_weights(state_dict):
    tmp_weights = OrderedDict()
    for name, params in iteritems(state_dict):
        tmp_weights[name.replace('module.', '')] = params
    return tmp_weights

class Model(object):
    def __init__(self, args, cycle, trial_dir_path):
        self.args = args
        self.log_path = None
        # self.log_write('', 'w')


    def log_write(self, log_str, mode='a+'):
        with open(self.log_path, mode) as writer:
            writer.write(log_str)

    def save_model(self, model, model_path):
        torch.save(model.state_dict(), model_path)

    def load_model(self, model, model_path):
        # model.load_state_dict(convert_weights(torch.load(model_path)))
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            model.load_state_dict(convert_weights(torch.load(model_path)))
        return model

    def models_save(self, models, score, last_best_score, keys_list=None):
        if score > last_best_score:
            if keys_list is None:
                keys_list = models.keys()
            for key in keys_list:
                self.save_model(models[key], self.save_path + "/BEST_{}_checkpoint.pt".format(key))

        return max(score, last_best_score)

    def models_load(self, models, key_list=None):
        if key_list is None:
            key_list = models.keys()
        for key in key_list:
            _ = self.load_model(models[key], self.save_path + "/BEST_{}_checkpoint.pt".format(key))

    def models_set_eval(self, models):
        for key in models.keys():
            models[key].eval()

    def models_set_train(self, models):
        for key in models.keys():
            models[key].train()

    def models_del(self, models):
        for key in models.keys():
            file_name = 'BEST_{}_checkpoint.pt'.format(key)
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)


class Controler(Model):
    def __init__(self, args, device, cycle, trial_dir_path, select_mode=-1, cnt_id=0):
        super(Controler, self).__init__(args, cycle, trial_dir_path)
        self.select_mode = select_mode
        if args.method_type in ['god_view_select_random', 'god_view_select_entropy']:
            self.save_path = build_dir('cycle_{}_mode{}'.format(cycle, select_mode), trial_dir_path)
        else:
            self.save_path = build_dir('cycle_{}_cnt_{}'.format(cycle, cnt_id), trial_dir_path)
        self.log_path = os.path.join(self.save_path, 'learner_record.log')
        # self.other_path = os.path.join(self.save_path, 'test_record.log')
        self.log_write('', 'w')

        self.device = device
        self.bcel_loss = nn.BCEWithLogitsLoss()

    def train_total(self, models, method, optimizers, schedulers, dataloaders, cycle, label_num=-1):
        print('>> Train a Model for use_random')
        best_score = 0.
        total_patience_cnt = 0
        for epoch in range(1, self.args.epochs + 1):
            for key in models.keys():
                models[key].train()

            if self.args.train_vae_module:
                models['vae_module'].reset_proto()
            for data in tqdm(dataloaders['labeled']):
                for key in optimizers.keys():
                    optimizers[key].zero_grad()
                labels, idx, loss_weight, _ = [x.to(self.device) for x in data[1:]]
                input_ids, input_type_ids, input_mask = [x.to(self.device) for x in data[0]]
                loss = 0
                inputs_list = [input_ids, input_type_ids, input_mask, None]
                clf_out, atten_out, encoder_cls = models['backbone'](inputs_list)
                criterion = nn.BCEWithLogitsLoss(weight=loss_weight, reduction='none' if 'loss_module' in models.keys() else 'mean')
                clf_loss = criterion(clf_out, labels)
                if 'loss_module' in models.keys():
                    new_clf_loss = torch.sum(clf_loss) / (clf_loss.size(0) * label_num)
                    lloss_cls = encoder_cls.detach()
                    lloss_out = models['loss_module'](lloss_cls)
                    lloss_loss = LossPredLoss(lloss_out, clf_out, MARGIN)
                    clf_loss = new_clf_loss + lloss_loss
                if 'mmc_module' in models.keys():
                    mmc_input = clf_out.detach()
                    mmc_input = torch.sigmoid(mmc_input)
                    mmc_input = torch.topk(mmc_input, label_num, dim=-1)[0]
                    mmc_input_sum = torch.sum(mmc_input, dim=-1) + 1e-10
                    mmc_input = mmc_input / mmc_input_sum.unsqueeze(-1)
                    label_cnt = torch.sum(labels, dim=-1).to(torch.long) - 1
                    label_cnt_neg_mask = (label_cnt < 0).to(torch.long)
                    label_cnt = label_cnt * (1 - label_cnt_neg_mask) + label_cnt_neg_mask * 0
                    label_cnt = label_cnt.to(torch.long)
                    mmc_output = models['mmc_module'](mmc_input)
                    mmc_criterion = nn.CrossEntropyLoss()
                    mmc_loss = mmc_criterion(mmc_output, label_cnt)
                    loss += mmc_loss

                loss += clf_loss
                loss.backward()
                for key in optimizers.keys():
                    optimizers[key].step()
            for key in schedulers.keys():
                schedulers[key].step()

            score = self.test(models, epoch, cycle, method, dataloaders)
            _ = self.models_save(models, score, best_score)
            if best_score >= score:
                total_patience_cnt += 1
                self.log_write('\nimpatience {}'.format(total_patience_cnt))
                if total_patience_cnt >= self.args.total_patience:
                    break
            else:
                best_score = score
                total_patience_cnt = 0
        self.models_load(models)

    def train_total_sep2(self, models, method, optimizers, schedulers, dataloaders, cycle, label_num=-1):
        print('>> Train a Model sep 2')
        best_score = 0.
        best_score2 = 0.
        second_phase = False
        total_patience_cnt = 0
        for epoch in range(1, self.args.epochs + 1):
            for key in models.keys():
                models[key].train()
            if second_phase:
                models['backbone'].eval()
            if self.args.train_vae_module:
                models['vae_module'].reset_proto()
            for data in tqdm(dataloaders['labeled']):
                for key in optimizers.keys():
                    optimizers[key].zero_grad()
                labels, idx, loss_weight, _ = [x.to(self.device) for x in data[1:]]
                input_ids, input_type_ids, input_mask = [x.to(self.device) for x in data[0]]
                loss = 0
                inputs_list = [input_ids, input_type_ids, input_mask, None]
                clf_out, atten_out, encoder_cls = models['backbone'](inputs_list)
                if self.args.train_vae_module:
                    loss2 = 0
                    module_cls = encoder_cls.detach()
                    dual_key_list = ['vae_module']
                    recon_x, latent_rep, recon_x_clf, _ = models['vae_module'](module_cls, labels=labels, weights=loss_weight)
                    cl_labels = self.get_cl_labels(labels, label_num, loss_weight).view(-1)
                    cl_labels_total = cl_labels.unsqueeze(-1)
                    proj_total = latent_rep.view(-1, latent_rep.shape[-1])
                    proj_total = F.normalize(proj_total, dim=-1)
                    batch_size = clf_out.shape[0] * label_num
                    mask, neg_mask = self.get_cl_mask(cl_labels_total, label_num, batch_size)
                    if not self.args.vae_cl_without_loss:
                        criterion_cl = SupConLoss()
                        cl_loss = criterion_cl(proj_total, mask, neg_mask, batch_size, self.device)
                        loss2 += cl_loss

                    criterion_mse = nn.MSELoss()
                    recon_mse_loss = criterion_mse(recon_x, module_cls)
                    criterion_bce = nn.BCEWithLogitsLoss(weight=loss_weight, reduction='mean')
                    recon_bce_loss = criterion_bce(recon_x_clf, labels)
                    if self.args.use_recon_bce_loss:
                        loss2 += recon_bce_loss
                    if self.args.use_recon_mse_loss:
                        loss2 += recon_mse_loss
                    loss += loss2
                if not second_phase:
                    criterion = nn.BCEWithLogitsLoss(weight=loss_weight, reduction='mean')
                    clf_loss = criterion(clf_out, labels)
                    loss += clf_loss
                loss.backward()
                if second_phase:
                    for key in dual_key_list:
                        optimizers[key].step()
                else:
                    for key in optimizers.keys():
                        optimizers[key].step()
            if second_phase:
                print('in second')
                for key in dual_key_list:
                    schedulers[key].step()
                score = self.test(models, epoch, cycle, method, dataloaders, test_dual=True)
                _ = self.models_save(models, score, best_score2, keys_list=dual_key_list)
                if best_score2 >= score:
                    total_patience_cnt += 1
                    self.log_write('\nimpatience {}'.format(total_patience_cnt))
                    print('impatience {}'.format(total_patience_cnt))
                    if total_patience_cnt >= self.args.total_patience:
                        break
                else:
                    best_score2 = score
                    total_patience_cnt = 0
            else:
                print('not in second')
                for key in schedulers.keys():
                    schedulers[key].step()
                score = self.test(models, epoch, cycle, method, dataloaders)
                _ = self.models_save(models, score, best_score)
                if best_score >= score:
                    total_patience_cnt += 1
                    self.log_write('\nimpatience {}'.format(total_patience_cnt))
                    print('impatience {}'.format(total_patience_cnt))
                    if total_patience_cnt >= self.args.total_patience :
                        second_phase = True
                        self.models_load(models, ['backbone'])
                        total_patience_cnt = 0
                        print('second phase')
                else:
                    best_score = score
                    total_patience_cnt = 0
        self.models_load(models)

    def test(self, models, epoch, cycle, method, dataloaders, log_file=None, other_file=None, test_dual=False, final_test=False):
        outputs_clf = [[], [], [], [], []]
        outputs_dis = [[], [], [], [], []]
        outputs_dual = [[], [], [], [], []]
        outputs_z = [[], [], [], [], []]
        self.models_set_eval(models)
        with torch.no_grad():
            for data in tqdm(dataloaders['test']):
                input_ids, input_type_ids, input_mask = [x.to(self.device) for x in data[0]]
                labels = data[1].to(self.device)
                out = [input_ids, input_type_ids, input_mask, None]

                clf_out, atten_out, encoder_cls = models['backbone'](out)
                outputs_clf = self.neaten_test_result(outputs_clf, clf_out, labels)
                if self.args.train_vae_module:
                    _, _, recon_x_clf, _ = models['vae_module'](encoder_cls)
                    outputs_dual = self.neaten_test_result(outputs_dual, recon_x_clf, labels)
        outputs_clf = self.concat_test_result(outputs_clf)
        if self.select_mode >= 0 and log_file is not None:
            write_2_file('\ncycle{} mode{}'.format(cycle, self.select_mode), log_file)
        n5_clf, hard_stop_mask = self.record_test_score(outputs_clf, cycle, epoch, log_file, other_file=other_file)
        if 'vae_module' in models.keys() and not final_test:
            outputs_dual = self.concat_test_result(outputs_dual)
            n5_dual, hard_stop_mask = self.record_test_score(outputs_dual, cycle, epoch, log_file,
                                                             other_file=other_file)
        if test_dual:
            return n5_dual
        return n5_clf


    def neaten_test_result(self, outputs, logits, labels):
        labels = labels.data.cpu().numpy()
        prob, pred = torch.topk(logits, self.args.topk)
        prob = torch.sigmoid(prob).data.cpu().numpy()
        pred = pred.data.cpu().numpy()
        logits = logits.data.cpu().numpy()
        pred_labels = np.zeros(logits.shape)
        pred_labels[logits >= 0] = 1
        outputs[0].append(labels)
        outputs[1].append(pred)
        outputs[2].append(prob)
        outputs[3].append(logits)
        outputs[4].append(pred_labels)
        return outputs

    def concat_test_result(self, outputs):
        outputs[0] = np.concatenate(outputs[0], axis=0)
        outputs[1] = np.concatenate(outputs[1], axis=0)
        outputs[2] = np.concatenate(outputs[2], axis=0)
        outputs[3] = np.concatenate(outputs[3], axis=0)
        outputs[4] = np.concatenate(outputs[4], axis=0)
        return outputs

    def record_test_score(self, outputs, cycle, epoch, log_file, other_file=None, xls_file=None):
        true_labels = label2list(outputs[0])
        p1, p3, p5, n3, n5 = base1_metric(true_labels, outputs[1], np.arange(self.args.label_num))
        log_str = '\n{}--{}  '.format(cycle, epoch) + '\t'.join(['%.5f'] * 5) + '\t'
        log_str = log_str % (p1, p3, p5, n3, n5)
        self.log_write(log_str)
        if log_file is not None:
            write_2_file(log_str, log_file)
        if other_file is not None:
            write_2_file(log_str, other_file)
        if xls_file is not None and cycle % 2 == 0:
            xls_log_str = '\n{} '.format(cycle) + '\t'.join(['%.5f'] * 2) + '\t'
            xls_log_str = xls_log_str % (p1, p3)
            write_2_file(xls_log_str, xls_file)
        ma_p, ma_r, ma_f1, mi_p, mi_r, mi_f1 = base2_metric(outputs[0], get_label(outputs[3], self.args.threshold))
        log_str = '\t'.join(['%.5f'] * 2)
        log_str = log_str % (ma_f1, mi_f1)
        self.log_write(log_str)
        if log_file is not None:
            write_2_file(log_str, log_file)
        if other_file is not None:
            write_2_file(log_str, other_file)

        p_score, r_score, f1_score = label_wise_prf1(outputs[0], outputs[4])

        f1_score = torch.tensor(f1_score)
        hard_stop_mask = torch.ones_like(f1_score)
        hard_stop_mask[f1_score >= self.args.hard_stop_threshold] = 0.0
        return n5, hard_stop_mask

    def get_cl_labels(self, labels, label_num, weight):
        cl_labels = torch.zeros_like(labels, dtype=torch.long)
        cl_labels.fill_(label_num)
        example_idx, label_idx = torch.where(labels>=0.5)
        cl_labels[example_idx, label_idx] = label_idx #.float()
        cl_labels = cl_labels * weight + label_num * (1 - weight)
        if self.args.cl_neg_mode == 2:
            example_idx, label_idx = torch.where(labels < 0.5)
            cl_neg_labels = label_idx + label_num
            cl_labels[example_idx, label_idx] = cl_neg_labels.float()
            cl_labels = cl_labels * weight + 2 * label_num * (1 - weight)
        cl_labels = cl_labels.to(torch.int)
        return cl_labels

    def get_cl_mask(self, cl_labels, label_num, batch_size):
        mask = torch.eq(cl_labels[:batch_size], cl_labels.T).float()
        if self.args.cl_neg_mode == 0:
            neg_idx, _ = torch.where(cl_labels[:batch_size] >= label_num)
            neg_mask = torch.ones_like(mask)
            neg_mask[neg_idx] = 0
            neg_mask[:, neg_idx] = 0
            mask *= neg_mask
        elif self.args.cl_neg_mode == 1:
            neg_idx, _ = torch.where(cl_labels[:batch_size] > label_num)
            neg_mask = torch.ones_like(mask)
            neg_mask[neg_idx] = 0
            neg_mask[:, neg_idx] = 0
            mask *= neg_mask
        elif self.args.cl_neg_mode == 2:
            neg_idx, _ = torch.where(cl_labels[:batch_size] >= label_num * 2)
            neg_mask = torch.ones_like(mask)
            neg_mask[neg_idx] = 0
            neg_mask[:, neg_idx] = 0
            mask *= neg_mask
        else:
            neg_mask = torch.ones_like(mask)
        return mask, neg_mask



