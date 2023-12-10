import pandas as pd
import numpy as np
import evaluate
import transformers

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer # TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
# from scipy.special import softmax

from dataclasses import dataclass, field
import logging
import os
# import argparse
# import torch
# import json
# import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@dataclass
class ModelConfig:
    model_path: str = "roberta-base"
    trust_remote_code = True,
    num_labels: int = 2
    num_hidden_layers: int = 18 # 12 def
    #ignore_mismatched_sizes = True, #dslim
    #classifier_dropout: float = 0.1
    #num_attention_heads: int = 16 # 12 def

    # hidden_act: str = "relu" # def:"gelu"; "relu", "silu" and "gelu_new"
    # position_embedding_type: str = "relative_key_query" # def:"absolute"; "relative_key", "relative_key_query"
    #hidden_dropout_prob: float = 0.3
    #attention_probs_dropout_prob: float = 0.25


@dataclass
class DatasetConfig:
    train_file: str = field(default=None, metadata={"help": "Path to train jsonl file"})
    test_file: str = field(default=None, metadata={"help": "Path to dev jsonl file"})


@dataclass
class TrainingArgsConfig(transformers.TrainingArguments):
    seed: int = 42
    output_dir: str = "experiments/"
    num_train_epochs: int = 3 # 10
    per_device_train_batch_size: int = 16 # 32
    per_device_eval_batch_size: int = 16 # 32
    auto_find_batch_size: bool = True
    logging_dir: str = "experiments/logs"
    logging_steps: int = 100
    run_name: str = 'exp'
    load_best_model_at_end: bool = True
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2

    weight_decay=0.01,
    learning_rate=2e-5,


def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True)


def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df


def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = 'nlp_hw2_taskA'

    parser = transformers.HfArgumentParser(
        (ModelConfig, DatasetConfig, TrainingArgsConfig)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("Model Arguments: ", model_args)
    print("Data Arguments: ", data_args)
    print("Training Arguments: ", training_args)

    dir = training_args.output_dir.split('/')[-1]
    training_args.run_name = f'{dir}_{training_args.num_train_epochs}ep_{training_args.per_device_train_batch_size}b_{training_args.per_device_eval_batch_size}b'
    transformers.set_seed(training_args.seed)
    model_path = model_args.model_path
    # random_seed = 42
    train_path =  data_args.train_file # For example 'subtaskA_train_multilingual.jsonl'
    test_path =  data_args.test_file # For example 'subtaskA_test_multilingual.jsonl'
    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}
    prediction_path = training_args.output_dir+'/predictions' # For example subtaskB_predictions.jsonl
    
    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    
    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    

    train_df, valid_df, test_df = get_data(train_path, test_path, training_args.seed)

    print('Loaded data.')
    print('................Starting preprocessing................')

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, truncate=True, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id#, ignore_mismatched_sizes = True
    )

    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        logger.info('\n................Start training................')
        # logger.info("Training...")
        logger.info("*** Train Dataset ***")
        logger.info(f"Number of samples: {len(tokenized_train_dataset)}")
        logger.info("*** Dev Dataset ***")
        logger.info(f"Number of samples: {len(tokenized_valid_dataset)}")

        trainer.train()

        logger.info("Training completed!")

        # save best model
        best_model_path = training_args.output_dir+'/best/'
        
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        
        trainer.save_model(best_model_path)


    if training_args.do_train:
        is_best=False
    else:
        is_best=True

    if training_args.do_predict:
        logger.info("\n................Start predicting................")

        if is_best:
            tokenizer = AutoTokenizer.from_pretrained(
                '/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/'+dir+'/best'
                )

            # load best model
            model = AutoModelForSequenceClassification.from_pretrained(
                '/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/'+dir+'/best',
                num_labels=len(label2id), id2label=id2label, label2id=label2id#, local_files_only=True
            )
        
        test_dataset = Dataset.from_pandas(test_df)
        tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})

        if is_best:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

        logger.info(f"*** Test Dataset ***")
        logger.info(f"Number of samples: {len(tokenized_test_dataset)}")

        # predictions, _, _ = trainer.predict(tokenized_test_dataset)
        predictions = trainer.predict(tokenized_test_dataset)
        
        # prob_pred = softmax(predictions.predictions, axis=-1) #???????????????????
        preds = np.argmax(predictions.predictions, axis=-1)
        metric = evaluate.load("bstrai/classification_report")
        results = metric.compute(predictions=preds, references=predictions.label_ids)
        
        logger.info("Predictions completed!")
        logging.info(results)

        predictions_df = pd.DataFrame({'id': test_df['id'], 'label': preds})
        
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        predictions_df.to_json(prediction_path+'/subtask_a_monolingual.jsonl', lines=True, orient='records')


