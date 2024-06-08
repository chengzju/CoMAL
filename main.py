import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from transformers import BertTokenizer, BertModel, set_seed
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
from config import *
from utils import *
from datasets import *
from networks import *
from losses import *
from controler import *
from selection_methods import *
from samplers import *
import warnings
from transformers import BertTokenizer, BertModel, set_seed, AdamW, get_linear_schedule_with_warmup, BertConfig
import time

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--gpuid', default='0', type=str)
parser.add_argument('--toy', action='store_true')
parser.add_argument('--toy_size', default=64, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--unlab_batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', help='path for the data folders')
parser.add_argument('--bert_path', default='./bert/bert-base-uncased')
parser.add_argument('--save_path')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--dynamic_split', action='store_true')  # stay false
parser.add_argument('--test_data_size', type=int, default=2000)  # useful when using dynamic_split
parser.add_argument('--maxlength', type=int, default=256)
parser.add_argument('--feature_layer', type=int, default=1) # use samplest 1
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--droprate', type=float, default=0.3)
parser.add_argument('--try_id', type=int, default=1)
parser.add_argument('--total_patience', type=int, default=10)
parser.add_argument('--train_size', type=int, default=40000)
parser.add_argument('--test_size', type=int, default=10000)
parser.add_argument('--cycles', default=9, type=int)
parser.add_argument('--mask_rate', type=float, default=0.15)
parser.add_argument("--method_type", type=str, default="lloss")
parser.add_argument('--drop_rate', type=float, default=0.0)
parser.add_argument('--sample_pair_num', type=int, default=1000)
parser.add_argument('--init_example_num', type=int, default=5000)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument("--adam_epsilon", default=1e-8, type=float)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--well_init', action='store_true')
parser.add_argument('--well_init_lower_bound', type=int, default=1)
parser.add_argument('--cl_neg_mode', type=int, default=0)
parser.add_argument('--hard_stop_threshold', type=float, default=0.9)
parser.add_argument('--soft_gamma', type=float, default=0.9)
parser.add_argument('--freeze_bert', action='store_true')
parser.add_argument('--freeze_layer_num', type=int, default=0)
parser.add_argument('--label_dis_init_mode', type=int, default=0)
parser.add_argument('--train_vae_module', action='store_true')
parser.add_argument('--use_recon_bce_loss', action='store_true')
parser.add_argument('--use_recon_mse_loss', action='store_true')
parser.add_argument('--vae_beta', default=0.0, type=float)
parser.add_argument('--vae_cl_without_loss', action='store_true')
parser.add_argument('--adaptive_b_mode', default=0, type=int)
parser.add_argument('--without_using_exist_well_init', action='store_true')
parser.add_argument('--proto_size', default=256, type=int)
parser.add_argument('--save_sub_tensor', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    method = args.method_type
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    if method in ['total', 'total_sample']:
        all_cycles = 1
    else:
        all_cycles = args.cycles
    root_dir = '{}_sd{}_cyc{}_id{}'.format(method, TRIALS, all_cycles, args.try_id)
    save_path = os.path.join(args.save_path, method)
    root_dir_path = build_dir(root_dir, save_path)
    for trial in range(1, 2):
        set_seed(args.seed)
        trial_dir_path = build_dir('trial_{}'.format(args.seed), root_dir_path)
        sampler_log_path = os.path.join(trial_dir_path, 'sampler_record.log')
        test_log_path = os.path.join(trial_dir_path, 'test_record.log')
        write_2_file('', sampler_log_path, 'w')
        write_2_file('', test_log_path, 'w')
        model_config = BertConfig.from_pretrained(args.bert_path)
        args.hidden_size = model_config.hidden_size * args.feature_layer
        label_num, label2id = load_label(args)
        args.label_num = label_num
        train_dataset, test_dataset, num_train, num_test = load_data(args, label2id)
        indices = list(range(num_train))
        random.shuffle(indices)
        if args.init_example_num > 0 :
            if args.well_init:
                labeled_set = train_dataset.get_well_init_indices()
            else:
                labeled_set = indices[:args.init_example_num]
        else:
            labeled_set = indices
        if method in ['total', 'total_sample']:
            labeled_set = indices
        unlabeled_set = [x for x in indices if x not in labeled_set]
        train_dataset.update_data(labeled_set,init_time=True)
        labeled_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=NUM_WORKER,
            pin_memory=True,
            sampler=SubsetRandomSampler(list(set(labeled_set)))
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=NUM_WORKER,
            pin_memory=True
        )
        unlabeled_loader = DataLoader(train_dataset,
                                      batch_size=args.unlab_batch_size,
                                      num_workers=NUM_WORKER,
                                      pin_memory=True,
                                      sampler=SubsetRandomSampler(indices)
                                      )


        for cycle in range(all_cycles):
            random.shuffle(unlabeled_set)
            dataloaders = {
                'labeled': labeled_loader,
                'test': test_loader
            }

            train_cnt = 1
            for cnt_id in range(train_cnt):
                models = {}
                optimizers = {}
                schedulers = {}
                # task learner model
                backbone = BackBone_No_GCN_No_Atten(args, label_num).to(device)
                models['backbone'] = backbone
                optim_backbone = AdamW(backbone.parameters(), lr=args.lr, eps=args.adam_epsilon)
                sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
                optimizers['backbone'] = optim_backbone
                schedulers['backbone'] = sched_backbone
                # sampler model
                if args.train_vae_module:
                    dual_module = MLP_VAE(args, args.hidden_size, device).to(device)
                    models['vae_module'] = dual_module
                    optim_module = AdamW(dual_module.parameters(), lr=1e-4, eps=args.adam_epsilon)
                    sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                    optimizers['vae_module'] = optim_module
                    schedulers['vae_module'] = sched_module
                if method in ['lloss']:
                    loss_module = LLoss_Module(args.hidden_size, label_num).to(device)
                    models['loss_module'] = loss_module
                    loss_optim = AdamW(loss_module.parameters(), lr=args.lr, eps=args.adam_epsilon)
                    loss_sched = lr_scheduler.MultiStepLR(loss_optim, milestones=MILESTONES)
                    optimizers['loss_module'] = loss_optim
                    schedulers['loss_module'] = loss_sched
                if method in ['mmc']:
                    mmc_module = MMC_Module(label_num).to(device)
                    models['mmc_module'] = mmc_module
                    mmc_optim = AdamW(mmc_module.parameters(), lr=args.lr, eps=args.adam_epsilon)
                    mmc_sched = lr_scheduler.MultiStepLR(mmc_optim, milestones=MILESTONES)
                    optimizers['mmc_module'] = mmc_optim
                    schedulers['mmc_module'] = mmc_sched
                controler = Controler(args, device, cycle, trial_dir_path, cnt_id=cnt_id)
                if method in ['core_set', 'badge', 'lloss', 'adaptive', 'mmc', 'cvirs', 'random']:
                    controler.train_total(models, method, optimizers, schedulers, dataloaders, cycle, label_num)
                else:
                    controler.train_total_sep2(models, method, optimizers, schedulers, dataloaders, cycle, label_num)

                score = controler.test(models, args.epochs, cycle, method, dataloaders, test_log_path, other_file=sampler_log_path, final_test=True)
                controler.models_del(models)

            if cycle == all_cycles - 1:
                controler.models_del(models)
                print('trial', args.seed, 'finish')
                break
            if method in ['core_set', 'badge']:
                query_set = indices
            else:
                query_set = list(set(unlabeled_set))
            random.shuffle(query_set)
            subset = query_set[:]

            if method in ['core_set', 'badge', 'lloss', 'adaptive', 'mmc', 'cvirs', 'random']:
                query_example_indices, query_label_indices, annotate_example_indices, annotate_label_indices = query_samples_other(
                    args, models, method, train_dataset, subset, device, label_num,
                    label_cardinality = train_dataset.get_label_Cardinality(),
                    labeled_subset = labeled_set
                )
            else:
                query_example_indices, query_label_indices, annotate_example_indices, annotate_label_indices = query_samples(
                    args, cycle, models, train_dataset, subset, device, label_num,
                    label_cardinality = train_dataset.get_label_Cardinality(),
                    labeled_subset=labeled_set,
                    record_path = trial_dir_path
                )
            train_dataset.update_data(query_example_indices, query_label_indices )
            labeled_set = train_dataset.get_labeled_set()
            unlabeled_set = [x for x in indices if x not in labeled_set]
            labeled_loader = DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=NUM_WORKER,
                                        pin_memory=True,
                                        sampler=SubsetRandomSampler(labeled_set)
                                        )
