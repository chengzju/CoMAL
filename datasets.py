from torch.utils.data import Dataset
import numpy as np
import torch
from collections import defaultdict
import os
import json
import math
import random


class ClfDataset(Dataset):
    def __init__(self, args, data, tokenizer, label2id):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.maxlength = args.maxlength
        self.all_labels_num = len(label2id)
        self.pair_wise_sampled = None
        self.gt_labels = None
        self.all_labeled_mask = None
        self.init()


    def init(self):
        if self.args.method_type == 'total' or self.args.method_type == 'total_sample':
            self.pair_wise_sampled = torch.ones(len(self.data), self.all_labels_num)
        else:
            self.pair_wise_sampled = torch.zeros(len(self.data), self.all_labels_num)
        self.all_labeled_mask = torch.zeros(len(self.data))
        self.gt_labels = torch.zeros(len(self.data), self.all_labels_num)
        for idx in range(len(self.data)):
            true_labels = list(set(self.data[idx]['label']) & set(self.label2id.keys()))
            true_labels_id = [self.label2id[x] for x in true_labels]
            self.gt_labels[idx][true_labels_id] = 1

    def get_well_init_indices(self):
        init_indices = []
        init_path = os.path.join(self.args.data_dir, 'well_init_indices_{}_{}.json'.format(self.args.init_example_num, self.args.well_init_lower_bound))
        root_path = os.path.join(self.args.data_dir, 'lab_2_ins_dict.json')
        if os.path.exists(root_path) and not self.args.without_using_exist_well_init:
            lab_2_ins_dict = json.load(open(root_path, 'r'))
        else:
            lab_2_ins_dict = defaultdict(list)
            for idx in range(self.__len__()):
                item = self.data[idx]  # {'id':int,'text':str,'labels':list}
                true_labels = list(set(item['label']) & set(self.label2id.keys()))
                for one_label in true_labels:
                    lab_2_ins_dict[one_label].append(idx)
            if not self.args.without_using_exist_well_init:
                json.dump(lab_2_ins_dict, open(root_path, 'w'), indent=2, ensure_ascii=False)
        for key in lab_2_ins_dict.keys():
            min_len = min(self.args.well_init_lower_bound, len(lab_2_ins_dict[key]))
            random_idx = random.sample(lab_2_ins_dict[key], min_len)
            init_indices.extend(random_idx)
        init_indices = list(set(init_indices))
        all_indices = list(range(self.__len__()))
        rest_indices = [x for x in all_indices if x not in init_indices]
        random.shuffle(rest_indices)
        rest_indices = rest_indices[:self.args.init_example_num - len(init_indices)]
        init_indices.extend(rest_indices)
        json.dump(init_indices, open(init_path, 'w'), indent=2, ensure_ascii=False)
        return init_indices


    def get_label_Cardinality(self):
        labed_cnt = torch.sum(self.all_labeled_mask)
        all_labed_pos_label_cnt = torch.sum(self.all_labeled_mask.unsqueeze(-1) * self.gt_labels)
        return all_labed_pos_label_cnt / labed_cnt

    def update_data(self, query_example_indices, query_label_indices=None, init_time=False):
        if query_label_indices is not None:
            ones = torch.ones(self.all_labels_num)
            for example_idx, label_idx in zip(query_example_indices, query_label_indices):
                one_data = self.__getitem__(example_idx)
                one_label = one_data[1][label_idx]
                if one_label == 0:
                    if init_time:
                        self.pair_wise_sampled[example_idx][label_idx] = 1
                    else:
                        self.pair_wise_sampled[example_idx][label_idx] = 1
                else:
                    self.pair_wise_sampled[example_idx][label_idx] = 1
                if torch.sum(ones - self.pair_wise_sampled[example_idx]) == 0:
                    self.all_labeled_mask[example_idx] = 1
        else:
            for example_idx in query_example_indices:
                ones = torch.ones(self.all_labels_num)
                self.pair_wise_sampled[example_idx] = ones
                self.all_labeled_mask[example_idx] = 1

    def get_labeled_set(self):
        labeled_set = []
        for idx, mask in enumerate(self.all_labeled_mask):
            if mask != 0:
                labeled_set.append(idx)
        return labeled_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx] # {'id':int,'text':str,'labels':list}

        true_labels = list(set(item['label']) & set(self.label2id.keys()))
        true_labels_id = [self.label2id[x] for x in true_labels]
        gt_labels = torch.zeros(self.all_labels_num)
        gt_labels[true_labels_id] = 1
        labels = gt_labels

        input_ids = item['input_ids']
        input_ids = [self.tokenizer.cls_token_id] + input_ids[:self.maxlength - 2] + [self.tokenizer.sep_token_id]
        input_type_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        # padding
        input_ids += [self.tokenizer.pad_token_id] * (self.maxlength - len(input_ids))
        input_type_ids += [self.tokenizer.pad_token_type_id] * (self.maxlength - len(input_type_ids))
        input_mask += [0] * (self.maxlength - len(input_mask))
        # to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        text_inputs = [input_ids, input_type_ids, input_mask]
        sampler_loss_weight = self.pair_wise_sampled[idx]

        inputs_list = [text_inputs, labels, idx, sampler_loss_weight, self.all_labeled_mask[idx]]
        return inputs_list





