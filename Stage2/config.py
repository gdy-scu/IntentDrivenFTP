# -*- coding: utf-8 -*-

class Config(object):
    # model configuration
    Pretrained_model_path = './pretrained_checkpoints/BERT-word/2023-01-06/checkpoint-349000/' # please use absolute path
    bert_out_dim = 256
    num_class = 17
    weight_decay = 1e-5
    learning_rate = 1e-4

    # train configuration
    batch_size = 512
    train_file_path = './data/example_data.csv'
    valid_file_path = './data/example_data.csv'
    intent_class_file = './vocab/intent_class.txt'
    vocab_path = './vocab/vocab_word.txt'
    model_save_path = './check_points/'




