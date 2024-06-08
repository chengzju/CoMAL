import os
import json
import random
import math
import numpy as np
from transformers import BertTokenizer
from datasets import *
import torch



def load_data(args, label2id):
    if 'rcv' in args.data_dir:
        train_file_name = 'clf_tidy_train_{}_{}_data.json'.format(args.train_size, args.maxlength)
        test_file_name = 'clf_tidy_test_{}_{}_data.json'.format(args.test_size, args.maxlength)
        train_data_path = os.path.join(args.data_dir, train_file_name)
        test_data_path = os.path.join(args.data_dir, test_file_name)
        train_data = json.load(open(train_data_path, 'r'))
        test_data = json.load(open(test_data_path, 'r'))
        print('in rcv')
    elif 'jd' in args.data_dir:
        train_file_name = 'clf_train_data_{}_{}.json'.format(args.test_size, args.maxlength)
        test_file_name = 'clf_test_data_{}_{}.json'.format(args.test_size, args.maxlength)
        train_data_path = os.path.join(args.data_dir, train_file_name)
        test_data_path = os.path.join(args.data_dir, test_file_name)
        train_data = json.load(open(train_data_path, 'r'))
        test_data = json.load(open(test_data_path, 'r'))
        print('in jd')
    else:
        if args.dynamic_split:
            train_save_path = os.path.join(args.data_dir,
                                           'clf_train_data_{}_{}.json'.format(args.maxlength, args.test_data_size))
            test_save_path = os.path.join(args.data_dir,
                                          'clf_test_data_{}_{}.json'.format(args.maxlength, args.test_data_size))
            if os.path.exists(train_save_path) and os.path.exists(test_save_path):
                print('use exist')
                train_data = json.load(open(train_save_path, 'r'))
                test_data = json.load(open(test_save_path, 'r'))
            else:
                train_file_name = 'clf_train_data_{}.json'.format(args.maxlength)
                test_file_name = 'clf_test_data_{}.json'.format(args.maxlength)
                train_data_path = os.path.join(args.data_dir, train_file_name)
                test_data_path = os.path.join(args.data_dir, test_file_name)
                train_data = json.load(open(train_data_path, 'r'))
                test_data = json.load(open(test_data_path, 'r'))
                total_data = train_data + test_data

                random.shuffle(total_data)
                test_data = total_data[:args.test_data_size]
                train_data = total_data[args.test_data_size:]
                json.dump(train_data, open(train_save_path, 'w'), indent=2, ensure_ascii=False)
                json.dump(test_data, open(test_save_path, 'w'), indent=2, ensure_ascii=False)
        else:
            train_file_name = 'clf_train_data_{}.json'.format(args.maxlength)
            test_file_name = 'clf_test_data_{}.json'.format(args.maxlength)
            train_data_path = os.path.join(args.data_dir, train_file_name)
            test_data_path = os.path.join(args.data_dir, test_file_name)
            train_data = json.load(open(train_data_path, 'r'))
            test_data = json.load(open(test_data_path, 'r'))
    if args.toy:
        train_data = train_data[:args.toy_size]
        test_data = test_data[:args.toy_size]
    print('train size', len(train_data), ',test size', len(test_data))

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_dataset = ClfDataset(
        args,
        data=train_data,
        tokenizer=tokenizer,
        label2id=label2id,
    )
    test_dataset = ClfDataset(
        args,
        data=test_data,
        tokenizer=tokenizer,
        label2id=label2id,
    )
    return train_dataset, test_dataset, len(train_data), len(test_data)

def load_label(args):
    label_freq_path = os.path.join(args.data_dir, 'label_freq.json')
    label_freq_desc = json.load(open(label_freq_path))
    label_weight = [x[1] for x in label_freq_desc]
    label_index = [x[0] for x in label_freq_desc]
    labels = label_index
    label2id = {j: i for i, j in enumerate(labels)}
    print('label num', len(labels))
    return len(labels), label2id

def label2list(label):
    outputs = [[] for _ in range(label.shape[0])]
    x,y = np.where(label==1)
    for xx,yy in zip(x,y):
        outputs[xx].append(yy)
    return outputs

def np_sigmoid(x):
    return 1/(1+np.exp(-x))

def get_label(prob, threshold):
    prob = prob.copy()
    pred = np.zeros(prob.shape)
    prob = np_sigmoid(prob)
    pred[prob > threshold] = 1
    return pred


def print_and_write_2_file(log_str, log_path, mode='a+'):
    print(log_str)
    write_2_file('\n'+log_str, log_path, mode)



def write_2_file(log_str, log_path, mode='a+'):
    with open(log_path, mode) as w:
        w.write(log_str)

def build_dir(dir_path, root_path = None):
    if root_path is not None:
        new_dir_path = os.path.join(root_path, dir_path)
    else:
        new_dir_path = dir_path
    if not os.path.isdir(new_dir_path):
        os.makedirs(new_dir_path)
    return new_dir_path

def list_to_str(one_list):
    txt = '\n'
    for one_cnt in one_list.to(torch.int).numpy():
        txt += str(one_cnt) + ','
    return txt


def get_H2(w, s, q):
    w_div_q = (w + 1e-10) / (q + 1e-6)
    s_div_q = (s + 1e-10) / (q + 1e-6)
    one_H2 = - torch.log2(w_div_q) * w_div_q - torch.log2(s_div_q) * s_div_q
    return one_H2

def get_H4(a, b, c, d, q):
    res = get_H2(b+c, a+d, q) + get_H2(b, c, b+c) * (b+c) / q + get_H2(a, d, a+d) * (a+d) / q
    return res


