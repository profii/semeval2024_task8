import pandas as pd
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from typing import List, Dict, Union
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.model_selection import train_test_split
from transformers import pipeline

from tqdm import tqdm
import os
import json

from args_parser import get_args


def get_dataset(path):
    """
    Generates dataset from jsonl file to DataFrame.

    Args:
    - path (str)

    Returns:
    - DataFrame: dataset.
    """

    with open(path, "r") as f:
        data = [json.loads(line) for line in f]


    return pd.DataFrame(data)


def generate_labels(df):
    """
    Generates labels [0, 1] according to label (index of the word split by whitespace where change happens),
    where 0 - human's text, 1 - machine's one.

    Args:
    - df (DataFrame): dataset

    Returns:
    - list(list(int)): lists of binary labels of every text,
    - list(list(str)): lists of words of every text.
    """
    
    labels = []
    words = []

    for _, text, label in df.values:
        # print(text)
        # print(label)
        tags = []

        for i in range(len(text.split())):
            if i < label:
                tags.append(0)
            else:
                tags.append(1)

            # print(i, tags[-1], token)

        labels.append(tags)
        words.append(text.split())

    return labels, words


def align_labels(texts, labels, max_length=510, label_all_tokens=True):
    """
    Aligns every binary label to tokens after applying tokenizer (-100 for special tokens).

    Args:
    - texts (list(list(str)): lists of words of every text,
    - labels (list(list(int))): lists of binary labels of every text,
    - max_length (int),
    - label_all_tokens (bool) 

    Returns:
    - dict: tokenized_inputs ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    """

    tokenized_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, is_split_into_words=True)
    l = []
    for i, label in enumerate(labels):
        # print(i, len(label), tokenized_inputs.word_ids(batch_index=i))

        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)

            previous_word_idx = word_idx

        l.append(label_ids)

    tokenized_inputs["labels"] = l

    return tokenized_inputs


def get_label(text, nlp):
    """
    Provides label using pipeline predictions.

    Args:
    - text (list(str)): lists of texts,
    - nlp (pipeline)

    Returns:
    - list: list of index of fisrt 1 in labels for every text.
    """

    label = []
    for t in tqdm(text):
        result = nlp(t)
        result = [int(i['entity'][-1:]) for i in result]
        if 1 in result:
            label.append(result.index(1))
        else:
            label.append(len(t))

    return label


class PairsDataset(Dataset):
    def __init__(self, x):
        self.y = x['labels']
        del x['labels']
        self.x = x

    def __getitem__(self, idx):
        assert idx <= len(self.x['input_ids']), (idx, len(self.x['input_ids']))
        item = {key: val[idx] for key, val in self.x.items()}
        item['labels'] = self.y[idx]

        return item

    @property
    def n(self):
        return len(self.x['input_ids'])

    def __len__(self):
        return self.n
    


if __name__ == "__main__":
    arg_parser = get_args()

    for arg in vars(arg_parser):
        print(arg, getattr(arg_parser, arg))

    torch.manual_seed(arg_parser.seed_val)
    torch.cuda.manual_seed_all(arg_parser.seed_val)

    model_name = arg_parser.model_name

    df = get_dataset(arg_parser.train_path)
    labels, words = generate_labels(df)
    df['labels'] = labels
    df['words'] = words

    X_train, X_val, y_train, y_val = train_test_split(df.words.values,
                                                  df.labels.values,
                                                  test_size=arg_parser.test_size,
                                                  random_state=arg_parser.seed_val)

    X_train.shape, X_val.shape, y_train.shape, y_val.shape

    print('Loaded dataset!')

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=arg_parser.add_prefix_space)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2).to('cuda')

    print(f'Loaded model {model_name}!')

    if arg_parser.max_length == 0:
        arg_parser.max_length = df.words.str.len().max() # 1120

    train_dataset = PairsDataset(align_labels(X_train.tolist(), y_train.tolist(), max_length=arg_parser.max_length))
    val_dataset = PairsDataset(align_labels(X_val.tolist(), y_val.tolist(), max_length=arg_parser.max_length))

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    

    args = TrainingArguments(output_dir=arg_parser.output_dir,
                             num_train_epochs=arg_parser.num_epochs,
                             per_device_train_batch_size=arg_parser.train_batch_size,
                             per_device_eval_batch_size=arg_parser.val_batch_size,
                             logging_dir=arg_parser.logging_dir,
                             logging_steps=arg_parser.logging_steps,
                             save_steps=arg_parser.save_steps,
                             evaluation_strategy = arg_parser.evaluation_strategy,
                             save_total_limit=arg_parser.save_total_limit,
                             save_strategy=arg_parser.save_strategy,
                            #  load_best_model_at_end=,
                            #  auto_find_batch_size=,
                            #  learning_rate=,
                            #  optim='adamw_torch',
                             )

    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        tokenizer = tokenizer,
        data_collator = data_collator,
        # compute_metrics=compute_metrics
    )

    print('Ready to train!')

    trainer.train()

    saved_name = '_'.join([model_name.split('-')[0], str(arg_parser.num_epochs)+'ep',
                           str(arg_parser.train_batch_size)+str(arg_parser.val_batch_size)+'b'])
    dir = arg_parser.output_dir + '/' + saved_name

    model.save_pretrained(dir)
    tokenizer.save_pretrained(dir+"_tok")

    print('Trained model is saved!')
    print('\nEvaluation:\n')


    df = get_dataset(arg_parser.test_path)
    labels, words = generate_labels(df)
    df['labels'] = labels
    df['words'] = words

    # test_dataset = PairsDataset(align_labels(df['words'].to_list(), df['labels'].to_list()))
    # pred = trainer.predict(test_dataset)


    saved_name = model_name+'_2ep_8b'
    model = AutoModelForTokenClassification.from_pretrained(arg_parser.output_dir+'/'+dir)
    tokenizer = AutoTokenizer.from_pretrained(arg_parser.output_dir+'/'+dir+"_tok")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    df_post = pd.DataFrame({"id": [id for id in df.id], "label": get_label(df.text, nlp)})


    file_name = os.path.basename('pred.jsonl')
    file_dirs = os.path.join("/content/drive/MyDrive/Colab Notebooks/NLP_HW/hw2/result", saved_name)
    os.makedirs(file_dirs, exist_ok=True)
    file_path = os.path.join(file_dirs, file_name)
    
    print('Saving into the file {file_path}')

    records = df_post.to_dict("records")
    with open(file_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print('A miracle happened ^-^/***')

