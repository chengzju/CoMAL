import os
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from collections import defaultdict
import re
from transformers import BertTokenizer


dataset_name = ['train', 'val', 'test']
data_name = 'text_{}'
label_name = 'label_{}'
root_path = './data/aapd'
bert_path = './bert/bert-base-uncased'

def raw_tokenize(sentence:str, sep='/SEP/', max_len=500):
    a=[token.lower() if token != sep else token for token in word_tokenize(sentence)
            if len(re.sub(r'[^\w]', '', token)) > 0]
    return a[:max_len], len(a)

def gen_label_file(dataset_name):
    label_cnt = defaultdict(int)
    for name in dataset_name:
        label_file = label_name.format(name)
        label_path = os.path.join(root_path, label_file)
        with open(label_path, 'r') as f:
            for i, line in enumerate(tqdm(f)):
                label_list = line.strip().split()
                for label in label_list:
                    label_cnt[label] += 1
    label_cnt_desc = sorted(label_cnt.items(), key=lambda x: x[1], reverse=True)
    freq_path = os.path.join(root_path, 'label_freq.json')
    json.dump(label_cnt_desc, open(freq_path, 'w'), indent=2, ensure_ascii=False)

def label_info(dataset_name):
    for name in dataset_name:
        max_label_num = 0
        min_label_num = 100
        mean_label_num = 0
        label_file = label_name.format(name)
        print(label_file)
        label_path = os.path.join(root_path, label_file)
        with open(label_path, 'r') as f:
            for i, line in enumerate(f):
                label_list = line.strip().split()
                max_label_num = max(max_label_num, len(label_list))
                min_label_num = min(min_label_num, len(label_list))
                mean_label_num = (mean_label_num * i + len(label_list))/(i+1)
        print(max_label_num, mean_label_num, min_label_num)

def gen_clf_data(dataset_name_sets, tokenizer, max_len):
    for new_name, old_name_set in dataset_name_sets.items():
        data = []
        last_size = 0
        for old_name in old_name_set:

            data_path = os.path.join(root_path, data_name.format(old_name))
            label_path = os.path.join(root_path, label_name.format(old_name))
            last_size = len(data)
            with open(data_path, 'r') as f:
                for i, line in enumerate(tqdm(f)):
                    d = {}
                    text = line.strip()
                    text_list, _ = raw_tokenize(text, max_len=max_len)

                    text = ' '.join(text_list)
                    input_toks = tokenizer.tokenize(text)

                    input_toks = input_toks[:max_len]
                    input_ids = tokenizer.convert_tokens_to_ids(input_toks)
                    d['input_ids'] = input_ids
                    data.append(d)

            with open(label_path, 'r') as f:
                for i, line in enumerate(f):
                    label_list = line.strip().split()
                    data[i+last_size]['label'] = label_list

        clf_save_file = 'clf_{}_data_{}.json'.format(new_name, max_len)
        clf_save_path = os.path.join(root_path, clf_save_file)
        print(new_name, len(data))
        json.dump(data, open(clf_save_path,'w'), indent=2, ensure_ascii=False)



def clf_data_info():
    cnt = 0
    file_name = 'clf_test_data_256.json'
    file_path = os.path.join(root_path, file_name)
    data = json.load(open(file_path, 'r'))
    print(type(data), len(data))
    for d in data:
        if 'label' not in d.keys():
            cnt += 1
    print(cnt)
            # print(1)
    # for k,v in data[0].items():
    #     print(k, type(v), v)


# def gen_mlm_data(dataset_name, tokenizer, max_len):
#     mHandler = MaskHandler(tokenizer, maxlength=max_len)
#     for name in dataset_name:
#         data = []
#         data_path = os.path.join(root_path, data_name.format(name))
#         with open(data_path, 'r') as f:
#             for i, line in tqdm(enumerate(f)):
#                 d = {}
#                 text = line.strip()
#                 text_list, _ = raw_tokenize(text, max_len=max_len)
#                 d['text'] = text_list
#                 data.append(d)
#         save_path = os.path.join(root_path, 'mlm_{}_data.json'.format(name))
#         mlm_data = mHandler.process(data, save_path)
#
#
# def mlm_data_info():
#     file_name = 'mlm_test_data.json'
#     file_path = os.path.join(root_path, file_name)
#     data = json.load(open(file_path, 'r'))
#     print(type(data), len(data))
#     for k,v in data[0].items():
#         print(k, type(v), v)


def main():
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    label_info(dataset_name)
    # gen_label_file(dataset_name)

    # dataset_name_sets = {
    #     # 'train':['train'],
    #     'test':['val', 'test']
    # }
    # len_list = [ 500,400,300,200]
    # for l in len_list:
    #     gen_clf_data(dataset_name_sets, tokenizer, max_len=l)
    clf_data_info()

    # gen_mlm_data(dataset_name, tokenizer, max_len=500)
    # mlm_data_info()

if __name__ == '__main__':
    main()

