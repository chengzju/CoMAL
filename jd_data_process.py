import os
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from collections import defaultdict
import re
from transformers import BertTokenizer
# from MaskHandler import MaskHandler
from collections import namedtuple
import csv
import random

dataset_name = ['train', 'val', 'test']
data_name = 'text_{}'
label_name = 'label_{}'
root_path = './data/jd'
bert_path = './bert/bert-base-uncased'
label_num = 38

def raw_tokenize(sentence:str, sep='/SEP/', max_len=500):
    a=[token.lower() if token != sep else token for token in word_tokenize(sentence)
            if len(re.sub(r'[^\w]', '', token)) > 0]
    return a[:max_len], len(a)


def read_jd_csv(input_file_path, output_file_paths,tokenizer, max_len, test_size):
    data = []
    read_cnt = 0
    words_cnt = 0
    right_label_cnt = 0
    pos_label_cnt = 0
    label_cnt = defaultdict(int)
    len_cnt_dict =defaultdict(int)
    max_len_flag = 0
    with open(input_file_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        label_names = headers[2: 2+label_num]

        # print(headers)
        # Row = namedtuple('Row')
        for row in reader:
            one_data = {}
            read_cnt += 1
            id = row[0]
            one_labels = row[2: 2+label_num]
            one_wrong_label_cnt = 0
            one_label_list = []
            for l,ln in zip(one_labels, label_names):
                if l=='0.0' or l =='1.0':
                    right_label_cnt += 1
                else:
                    one_wrong_label_cnt += 1
                if l=='1.0':
                    pos_label_cnt += 1
                    label_cnt[ln] += 1
                    one_label_list.append(ln)
            # print(one_label_list)
            one_data['label'] = one_label_list
            # if one_wrong_label_cnt > 0:
            #     print(id, one_wrong_label_cnt)

            # print(len(one_labels), one_labels)



            text = row[1].strip()
            words = text.split()
            words_cnt += len(words)
            text_list, _ = raw_tokenize(text, max_len=max_len)
            text = ' '.join(text_list)
            input_toks = tokenizer.tokenize(text)
            input_toks = input_toks[:max_len]
            input_ids = tokenizer.convert_tokens_to_ids(input_toks)
            one_data['input_ids'] = input_ids
            # print('input toks len', len(input_ids), input_ids)
            len_cnt_dict[len(input_ids)] += 1
            if len(input_ids) > max_len_flag:
                max_len_flag = len(input_ids)
            data.append(one_data)

            # print(row)

            # if read_cnt > 10:
            #     break
    print('max len flag', max_len_flag)
    mean_words_cnt = words_cnt / read_cnt
    print(mean_words_cnt, read_cnt)
    print(right_label_cnt, right_label_cnt/label_num)
    print(pos_label_cnt, pos_label_cnt / read_cnt)
    print('data cnt', len(data))
    label_cnt_desc = sorted(label_cnt.items(), key=lambda x: x[1], reverse=True)
    # print(label_cnt_desc)
    len_cnt_desc = sorted(len_cnt_dict.items(), key=lambda x: x[0], reverse=True)
    print(len_cnt_desc)
    freq_path = os.path.join(root_path, 'label_freq.json')
    json.dump(label_cnt_desc, open(freq_path, 'w'), indent=2, ensure_ascii=False)
    # print(data[0])
    random.shuffle(data)
    # print(data[0])
    test_data = data[:test_size]
    train_data = data[test_size:]
    # train_save_file = 'clf_{}_data_{}.json'.format(new_name, max_len)
    # clf_save_path = os.path.join(root_path, clf_save_file)
    # print(new_name, len(data))
    print(len(train_data)+ len(test_data))
    json.dump(train_data, open(output_file_paths[0], 'w'), indent=2, ensure_ascii=False)
    json.dump(test_data, open(output_file_paths[1], 'w'), indent=2, ensure_ascii=False)


def main():
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    input_file_path = os.path.join(root_path, 'jd-dataset-topics-Human-Review.csv')
    max_len = 128
    test_size = 1500
    train_output_file_path = os.path.join(root_path, 'clf_train_data_{}_{}.json'.format(test_size, max_len))
    test_output_file_path = os.path.join(root_path, 'clf_test_data_{}_{}.json'.format(test_size, max_len))
    output_file_paths = [train_output_file_path, test_output_file_path]
    read_jd_csv(input_file_path, output_file_paths, tokenizer, max_len, test_size)


if __name__ == '__main__':
    main()

