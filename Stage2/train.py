# -*- coding: utf-8 -*-
import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertTokenizer
from utils import ATCDataset, convert_text_to_ids, seq_padding
from config import Config
from IID_model import BERT4IntentDetection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class transformers_bert_intent_classification(object):
    def __init__(self):
        self.config = Config()
        self.device_setup()

    def device_setup(self):
        self.freezeSeed()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(self.device)

        self.tokenizer = BertTokenizer(vocab_file=self.config.vocab_path, model_max_length=512)

        print("Init Model...")
        self.model = BERT4IntentDetection(Config)

        print(self.model)

        self.model_save_path = self.config.model_save_path + '/' + str(
            datetime.date.today())

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.log_file = open(os.path.join(self.model_save_path, 'train.log'), 'w')

        self.model.to(self.device)

    def model_setup(self):
        weight_decay = self.config.weight_decay
        learning_rate = self.config.learning_rate

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        self.criterion = nn.BCELoss()

    def get_data(self):
        batch_size = self.config.batch_size

        train_set = ATCDataset(self.config.train_file_path, self.config)
        valid_set = ATCDataset(self.config.valid_file_path, self.config)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, valid_loader

    def train_an_epoch(self, iterator):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0

        for i, batch in enumerate(iterator):
            label = batch["label"]
            text = batch["text"]
            input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, text)
            input_ids = seq_padding(self.tokenizer, input_ids)
            token_type_ids = seq_padding(self.tokenizer, token_type_ids)
            input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.float()

            input_ids, token_type_ids, label = input_ids.to(self.device), token_type_ids.to(self.device), label.to(
                self.device)

            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)

            y_pred_prob = output[1]
            y_pred_label = (y_pred_prob > 0.5).int()

            loss = self.criterion(y_pred_prob.view(-1, self.config.num_class), label.view(-1, self.config.num_class))

            acc = sum([y_pred_label[i].cpu().numpy().tolist() == label[i].cpu().numpy().tolist() for i in
                       range(len(y_pred_prob))])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            if i % 200 == 0:
                print("current loss:", epoch_loss / (i + 1), "\t", "current acc:", epoch_acc / ((i + 1) * len(label)))

        return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)

    def cal_metrics(self, y_true, y_pred):
        TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
        FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

        Acc = (TP + TN) / (TP + FP + FN + TN)
        Prec = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * Prec * Recall / (Prec + Recall)

        return Acc, Prec, Recall, F1

    def evaluate(self, iterator):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        y_pred = []
        y_true = []
        with torch.no_grad():
            for _, batch in enumerate(iterator):
                label = batch["label"]
                text = batch["text"]
                input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, text)
                input_ids = seq_padding(self.tokenizer, input_ids)
                token_type_ids = seq_padding(self.tokenizer, token_type_ids)

                input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.float()
                input_ids, token_type_ids, label = input_ids.to(self.device), token_type_ids.to(self.device), label.to(
                    self.device)
                output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)

                y_pred_prob = output[1]
                y_pred_label = (y_pred_prob > 0.5).int()

                loss = self.criterion(y_pred_prob.view(-1, self.config.num_class),
                                      label.view(-1, self.config.num_class))

                acc = sum([y_pred_label[i].cpu().numpy().tolist() == label[i].cpu().numpy().tolist() for i in
                           range(len(y_pred_prob))])

                y_true.extend(label.cpu().numpy())
                y_pred.extend(y_pred_label.cpu().numpy())

                epoch_loss += loss.item()
                epoch_acc += acc

        acc = accuracy_score(np.array(y_true), np.array(y_pred))
        prec = precision_score(np.array(y_true), np.array(y_pred), average='weighted')
        recall = recall_score(np.array(y_true), np.array(y_pred), average='weighted')
        f1 = f1_score(np.array(y_true), np.array(y_pred), average='weighted')

        print("Acc:{}, Precision:{}, Recall:{}, F1 Score:{}".format(acc, prec, recall, f1))
        print("Acc:{}, Precision:{}, Recall:{}, F1 Score:{}".format(acc, prec, recall, f1), file=self.log_file)
        return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)

    def train(self, epochs):
        self.model_setup()
        train_loader, valid_loader = self.get_data()

        best_acc = 0.
        for i in range(epochs):
            train_loss, train_acc = self.train_an_epoch(train_loader)
            print("train loss: ", train_loss, "\t", "train acc:", train_acc)
            valid_loss, valid_acc = self.evaluate(valid_loader)
            print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
            print("Best acc:", best_acc)

            if valid_acc > best_acc:
                best_acc = valid_acc
                self.save_model("{}/bert_best_acc{}".format(self.model_save_path, valid_acc) + ".pt")
                print("model saved...")

    def save_model(self, model_save_path):
        vocab_save_path = os.path.join(self.model_save_path, 'vocab.txt')

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), model_save_path)

        self.tokenizer.save_vocabulary(vocab_save_path)
        print("model saved...")

    def predict(self, sentence):
        self.model.eval()
        input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, sentence)
        input_ids = seq_padding(self.tokenizer, [input_ids])
        token_type_ids = seq_padding(self.tokenizer, [token_type_ids])
        input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
        input_ids, token_type_ids = input_ids.to(self.device), token_type_ids.to(self.device)
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
        y_pred_prob = output[1]
        y_pred_label = (y_pred_prob > 0.5).int()

        return y_pred_label

    def freezeSeed(self):
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)


if __name__ == '__main__':
    classifier = transformers_bert_intent_classification()
    classifier.train(10)
    print(classifier.predict("上 到 八 千 九 保 持 国 航 九 六 八"))
    print(classifier.predict("东 方 五 三 八 三 右 转 飞 kakmi"))
