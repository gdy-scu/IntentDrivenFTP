# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ATCDataset(Dataset):
    def __init__(self, path_to_file, config):
        self.config = config
        self.intent2idx = {}
        self.idx2intent = {}

        self.load_intent()

        df = pd.read_csv(path_to_file)

        self.dataset = self.generate_label(df)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "text"]
        label = self.dataset.loc[idx, "label"]
        sample = {"text": text, "label": label}
        return sample

    def load_intent(self):
        with open(self.config.intent_class_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                intent = line.strip().split()[0]
                index = int(line.strip().split()[1])
                self.intent2idx[intent] = index
                self.idx2intent[index] = intent

    def generate_label(self, df):
        self.num_class = len(self.intent2idx)

        df['label'] = ''
        df['label'] = df['label'].astype(object)

        for row in df.index:
            intents = df.loc[row]['intents']
            df.at[row, 'label'] = np.zeros(self.num_class)
            for i in intents.strip().split("#"):
                idx = self.intent2idx[i]
                df.loc[row]['label'][idx] = 1

        return df


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
