import os
import math
import datetime
import tokenizers
import transformers
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoConfig,
    AutoModelForMaskedLM,
    BertModel,
    BertForMaskedLM,
    BertConfig,
)


def tokenize_function(examples):
    s = tokenizer(examples["text"])
    return s

# block_size = tokenizer.model_max_length
block_size = 128

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in
                             ["attention_mask", "input_ids", "token_type_ids"]}  # examples.keys()}
    total_length = len(concatenated_examples['input_ids'])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


train_csv = "./data/train.csv"
valid_csv = "./data/valid.csv"
tokenizer_vocab = "./vocab/vocab_word.txt"

tokenizer = BertTokenizer(vocab_file=tokenizer_vocab, model_max_length=512)

model = BertForMaskedLM(BertConfig(
    vocab_size=len(tokenizer),
    num_hidden_layers=4,
    hidden_size=256,
    num_attention_heads=4)
)
save_path = f"./pretrained_checkpoints/BERT-word/{datetime.date.today()}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

datasets = load_dataset('csv', data_files={'train': train_csv, 'validation': valid_csv})
tokenized_datasets = datasets.map(tokenize_function, batched=True,
                                  num_proc=4,
                                  remove_columns=["lang", "key", "text", "snr", "context", "duration", "time",
                                                  "readable", "intents", "slu_label", "role", "callsign"])

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=8000,
    num_proc=4,
)

training_args = TrainingArguments(
    save_path,
    evaluation_strategy="epoch",
    # learning_rate=2e-5,
    learning_rate=1e-4,
    weight_decay=0.01,
    per_device_train_batch_size=36,
    # per_device_train_batch_size=2,
    num_train_epochs=1000,
    save_total_limit=10,
    push_to_hub=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
