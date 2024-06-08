import os
import json
from tqdm import tqdm
from collections import defaultdict
from transformers import BertTokenizer
# from MaskHandler import MaskHandler
import torch
import random

train_file_name = 'lyrl2004_tokens_train.dat'
test_file_name = 'lyrl2004_tokens_test_pt{}.dat'
label_file_name = 'rcv1-v2.topics.qrels'
root_path = './data/rcv1-v2'
bert_path = './bert/bert-base-uncased'


def get_tidy_text(old_text_list):
    new_text_list = []
    for text in old_text_list:
        if len(new_text_list) <= 0 or text != new_text_list[-1]:
            new_text_list.append(text)
    return new_text_list

def sub_data_info(file_name_list):
    res = []
    for file_idx, file_name in enumerate(file_name_list):
        one_res = []
        file_path = os.path.join(root_path, file_name)
        with open(file_path, 'r') as f:
            for i, line in enumerate(tqdm(f)):
                if line.startswith('.I'):
                    idx = line.strip().split()[-1]
                    one_res.append(idx)
        res.append(one_res)
        print(len(one_res))

    return res

def sub_label_info(sub_idx_lists,  label_dict, label_cnt_desc):
    labels_list = [x[0] for x in label_cnt_desc]
    label_to_id = {}
    label_num = len(labels_list)
    for i, x in enumerate(labels_list):
        label_to_id[x] = i
    res = []
    for one_sub in sub_idx_lists:
        one_res = torch.zeros(label_num)
        for one_ins_idx in one_sub:
            one_ins_label_list = label_dict[one_ins_idx]
            for one_label in one_ins_label_list:
                one_res[label_to_id[one_label]] += 1
        # print(one_res)
        res.append(one_res)
    for label_id in  range(label_num):
        print(res[1][label_id],res[2][label_id],res[3][label_id],res[4][label_id])

    # for one_list in sub_idx_lists:
    return


def gen_data_file(file_name_list):
    for file_idx, file_name in enumerate(file_name_list):
        new_text = []
        new_tidy_text = []
        data_list = []
        data_tidy_list = []
        last_data = {}
        last_tidy_data = {}

        file_path = os.path.join(root_path, file_name)
        with open(file_path, 'r') as f:
            for i, line in enumerate(tqdm(f)):
                if line.startswith('.I'):
                    idx = line.strip().split()[-1]
                    last_data = {}
                    last_tidy_data = {}
                    last_data['id'] = idx
                    last_tidy_data['id'] = idx
                    last_data['text'] = []
                    last_tidy_data['text'] = []
                    new_text = []
                    new_tidy_text = []
                elif line.startswith('.W'):
                    continue
                elif line.strip() == '':
                    if last_data == {}:
                        continue
                    last_data['text'] = new_text
                    last_tidy_data['text'] = new_tidy_text
                    data_list.append(last_data.copy())
                    data_tidy_list.append(last_tidy_data.copy())
                else:
                    one_word_list = line.strip().split()
                    tidy_one_word_list = get_tidy_text(one_word_list)
                    new_text.extend(one_word_list)
                    if len(new_tidy_text) == 0:
                        new_tidy_text.extend(tidy_one_word_list)
                    else:
                        if new_tidy_text[-1] == tidy_one_word_list[0]:
                            new_tidy_text.extend(tidy_one_word_list[1:])
                        else:
                            new_tidy_text.extend(tidy_one_word_list)
                    # new_tidy_text.extend(tidy_one_word_list)
        output_file_name = 'new_data{}.json'.format(file_idx)
        output_tidy_file_name = 'new_tidy_data{}.json'.format(file_idx)
        output_file_path = os.path.join(root_path, output_file_name)
        output_tidy_file_path = os.path.join(root_path, output_tidy_file_name)
        json.dump(data_list, open(output_file_path, 'w'), indent=2, ensure_ascii=False)
        json.dump(data_tidy_list, open(output_tidy_file_path, 'w'), indent=2, ensure_ascii=False)


def gen_label_file(label_file_path):
    label_dict = defaultdict(list)
    label_cnt = defaultdict(int)
    with open(label_file_path, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            items = line.strip()
            if items == '':
                continue
            items = items.split()
            label_dict[items[1]].append(items[0])
            label_cnt[items[0]] += 1
    save_label_file = 'new_label.json'
    save_label_path = os.path.join(root_path, save_label_file)
    json.dump(label_dict, open(save_label_path, 'w'), indent=2, ensure_ascii=False)

    label_cnt_desc = sorted(label_cnt.items(), key=lambda x: x[1], reverse=True)
    freq_path = os.path.join(root_path, 'label_freq.json')
    json.dump(label_cnt_desc, open(freq_path, 'w'), indent=2, ensure_ascii=False)
    return label_dict, label_cnt_desc


def gen_total_data(tidy_train_size, tidy_test_size):
    data_file_name = 'new_data{}.json'
    tidy_data_file_name = 'new_tidy_data{}.json'
    data = {}
    tidy_data = {}
    train_data = {}
    tidy_train_data = {}
    test_data = {}
    tidy_test_data = {}
    for i in range(5):
        data_file_path = os.path.join(root_path, data_file_name.format(i))
        tidy_data_file_path = os.path.join(root_path, tidy_data_file_name.format(i))
        data_patch = json.load(open(data_file_path, 'r'))
        tidy_data_patch = json.load(open(tidy_data_file_path, 'r'))
        for one_data in data_patch:
            data[one_data['id']] = one_data
        for one_data in tidy_data_patch:
            tidy_data[one_data['id']] = one_data
        if i == 0:
            for one_data in data_patch:
                train_data[one_data['id']] = one_data
            for one_data in tidy_data_patch:
                tidy_train_data[one_data['id']] = one_data
        else:
            for one_data in data_patch:
                test_data[one_data['id']] = one_data
            for one_data in tidy_data_patch:
                tidy_test_data[one_data['id']] = one_data
    train_id_set = train_data.keys()
    test_id_set = test_data.keys()

    label_file_name = 'new_label.json'
    label_file_path = os.path.join(root_path, label_file_name)
    labels = json.load(open(label_file_path, 'r'))
    for k, v in tqdm(labels.items()):

        if k in train_id_set:
            train_data[k]['label'] = v
            tidy_train_data[k]['label'] = v
        else:
            test_data[k]['label'] = v
            tidy_test_data[k]['label'] = v
    data_list = []
    tidy_data_list = []

    for k,v in data.items():
        data_list.append(v)
    for k,v in tidy_data.items():
        tidy_data_list.append(v)

    print(len(data_list), len(tidy_data_list))
    output_file_name = 'total_data.json'
    output_tidy_file_name = 'total_tidy_data.json'
    output_file_path = os.path.join(root_path, output_file_name)
    output_tidy_file_path = os.path.join(root_path, output_tidy_file_name)
    # json.dump(data_list, open(output_file_path, 'w'), indent=2, ensure_ascii=False)
    # json.dump(tidy_data_list, open(output_tidy_file_path, 'w'), indent=2, ensure_ascii=False)

    train_data_list = []
    tidy_train_data_list = []
    for k,v in train_data.items():
        train_data_list.append(v)
    for k,v in tidy_train_data.items():
        tidy_train_data_list.append(v)

    print(len(train_data_list), len(tidy_train_data_list))



    test_data_list = []
    tidy_test_data_list = []
    for k, v in test_data.items():
        test_data_list.append(v)
    for k, v in tidy_test_data.items():
        tidy_test_data_list.append(v)
    # print(tidy_test_data_list[0])
    random.shuffle(tidy_test_data_list)
    if len(tidy_train_data_list) < tidy_train_size:
        tidy_train_data_list.extend(tidy_test_data_list[-(tidy_train_size - len(tidy_train_data_list)):])
    tidy_train_data_list2 = tidy_train_data_list[:30000]
    tidy_train_data_list3 = tidy_train_data_list[:23149]
    tidy_test_data_list = tidy_test_data_list[:tidy_test_size]
    # print(tidy_test_data_list[0])
    print(len(train_data_list), len(tidy_train_data_list), len(tidy_train_data_list2),len(tidy_train_data_list3))
    print(len(test_data_list), len(tidy_test_data_list))


    # output_file_name = 'train_data.json'
    # output_tidy_file_name = 'train_tidy_data.json'
    # output_file_path = os.path.join(root_path, output_file_name)
    # output_tidy_file_path = os.path.join(root_path, output_tidy_file_name)
    # json.dump(train_data_list, open(output_file_path, 'w'), indent=2, ensure_ascii=False)
    # json.dump(tidy_train_data_list, open(output_tidy_file_path, 'w'), indent=2, ensure_ascii=False)
    # output_file_name = 'test_data.json'
    # output_tidy_file_name = 'test_tidy_data.json'
    # output_file_path = os.path.join(root_path, output_file_name)
    # output_tidy_file_path = os.path.join(root_path, output_tidy_file_name)
    # json.dump(test_data_list, open(output_file_path, 'w'), indent=2, ensure_ascii=False)
    # json.dump(tidy_test_data_list, open(output_tidy_file_path, 'w'), indent=2, ensure_ascii=False)
    return train_data_list, tidy_train_data_list, test_data_list, tidy_test_data_list, tidy_train_data_list2, tidy_train_data_list3



def total_data_info():
    file_name = 'total_tidy_data.json'
    file_path = os.path.join(root_path, file_name)
    data = json.load(open(file_path, 'r'))
    print(type(data), len(data))
    for k,v in data[0].items():
        print(k, type(v), v)


def gen_clf_data2(data, save_file, tokenizer, max_length):
    # file_path = os.path.join(root_path, file_name)
    # data = json.load(open(file_path, 'r'))
    new_data = []
    max_len = 0
    for item in tqdm(data):
        d = {}
        id = item['id']
        text_list = item['text'][:max_length]
        label = item['label']
        text = ' '.join(text_list)

        input_toks = tokenizer.tokenize(text)
        max_len = max(max_len, len(input_toks))
        input_toks = input_toks[:max_length]
        input_ids = tokenizer.convert_tokens_to_ids(input_toks)
        d['input_ids'] = input_ids
        d['label'] = label
        new_data.append(d)
    # print(max_len)

    save_path = os.path.join(root_path, save_file)
    json.dump(new_data, open(save_path, 'w'), indent=2, ensure_ascii=False)

def gen_clf_data(file_name, save_file, tokenizer, max_length):
    file_path = os.path.join(root_path, file_name)
    data = json.load(open(file_path, 'r'))
    new_data = []
    max_len = 0
    for item in tqdm(data):
        d = {}
        id = item['id']
        text_list = item['text'][:max_length]
        label = item['label']
        text = ' '.join(text_list)

        input_toks = tokenizer.tokenize(text)
        max_len = max(max_len, len(input_toks))
        input_toks = input_toks[:max_length]
        input_ids = tokenizer.convert_tokens_to_ids(input_toks)
        d['input_ids'] = input_ids
        d['label'] = label
        new_data.append(d)
    # print(max_len)

    save_path = os.path.join(root_path, save_file)
    json.dump(new_data, open(save_path, 'w'), indent=2, ensure_ascii=False)


def clf_data_info():
    file_name = 'clf_tidy_data.json'
    file_path = os.path.join(root_path, file_name)
    data = json.load(open(file_path, 'r'))
    print(type(data), len(data))
    for k,v in data[0].items():
        print(k, type(v), v)

def mlm_data_info():
    file_name = 'mlm_tidy_data.json'
    file_path = os.path.join(root_path, file_name)
    data = json.load(open(file_path, 'r'))
    print(type(data), len(data))
    for k,v in data[0].items():
        print(k, type(v), v)



def main():
    file_name_list = [train_file_name] + [test_file_name.format(i) for i in range(4)]
    label_file_path = os.path.join(root_path, label_file_name)
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # gen_data_file(file_name_list)
    # sub_res = sub_data_info(file_name_list)
    label_dict, label_cnt_desc = gen_label_file(label_file_path)
    print(label_cnt_desc)
    # sub_label_info(sub_res, label_dict, label_cnt_desc)
    tidy_train_size = 40000
    tidy_test_size = 10000
    train_data_list, tidy_train_data_list, test_data_list, tidy_test_data_list, tidy_train_data_list2, tidy_train_data_list3\
        = gen_total_data(tidy_train_size, tidy_test_size)
    total_data_info()
    text_len = 256
    gen_clf_data2(tidy_train_data_list, 'clf_tidy_train_{}_{}_data.json'.format(tidy_train_size, text_len), tokenizer, text_len)
    gen_clf_data2(tidy_train_data_list2, 'clf_tidy_train_{}_{}_data.json'.format(30000, text_len), tokenizer,
                  text_len)
    gen_clf_data2(tidy_train_data_list3, 'clf_tidy_train_{}_{}_data.json'.format(23149, text_len), tokenizer,
                  text_len)
    gen_clf_data2(tidy_test_data_list, 'clf_tidy_test_{}_{}_data.json'.format(tidy_test_size, text_len), tokenizer, text_len)




if __name__ == '__main__':
    main()

