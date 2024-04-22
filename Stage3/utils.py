# coding=utf-8
import torch
import json
import numpy as np
import time
import os


def convert_text_to_ids(tokenizer, text, max_len=100):
    if isinstance(text, str):
        tokenized_text = tokenizer.encode_plus(text, max_length=max_len, add_special_tokens=True, truncation=True)
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]
    elif isinstance(text, list):
        input_ids = []
        token_type_ids = []
        for t in text:
            tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True, truncation=True)
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])
    else:
        print("Unexpected input")
    return input_ids, token_type_ids


def seq_padding(tokenizer, X):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if len(X) <= 1:
        return torch.tensor(X)
    L = [len(x) for x in X]
    ML = max(L)
    X = torch.Tensor([x + [pad_id] * (ML - len(x)) if len(x) < ML else x for x in X])
    return X


def dict_to_obj(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, dict_to_obj(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                    type(j)(dict_to_obj(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top


def load_config_from_json(json_path='config.json'):
    with open(json_path, 'r') as fr:
        data = json.load(fr)
    return dict_to_obj(data)


def print_attrs(obj):
    print("****PRINT ATTRS****")
    ATTRS = ''
    for name in dir(obj):
        if not name.startswith('__'):
            ATTRS += "{}:{}\n".format(name, getattr(obj, name))

    print(ATTRS)
    return ATTRS


def file_print(info, logfilename='log.txt', savepath=None, debug=True):
    t = time.localtime(int(time.time()))
    ts = time.strftime('%m-%d %H:%M:%S ', t)
    if savepath is not None:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        fullfilename = os.path.join(savepath, logfilename)
    else:
        fullfilename = logfilename

    if debug:
        print(ts + info)

    with open(fullfilename, 'a') as f:
        f.write(ts + info + '\n')


def convert2binfromlist(bool_list):
    l = len(bool_list)
    res = 0
    for i in range(l):
        if bool_list[i] == 1:
            res += 2 ** (l - i - 1)
    return res