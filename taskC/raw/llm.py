import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
# from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.model_selection import train_test_split
# from transformers import pipeline
from datasets import load_metric
from peft import LoraConfig
import logging

from args_parser import get_args
from pathlib import Path
from tqdm import tqdm
import os
import json
import wandb

from transformers.trainer_callback import TrainerState
import transformers

from dataclasses import dataclass, field
from typing import List#, Any, Optional
import glob


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()



@dataclass
class ModelConfig:
    model_path: str = "allenai/longformer-base-4096"
    trust_remote_code: bool = True,
    ignore_mismatched_sizes: bool = True, #dslim
    # num_labels: int = 2 # YES!!!
    # num_hidden_layers: int = 26 # 12 def
    # # classifier_dropout: float = 0.1


@dataclass
class DatasetConfig:
    train_file: str = field(default=None, metadata={"help": "Path to train jsonl file"})
    dev_file: str = field(default=None, metadata={"help": "Path to dev jsonl file"})
    test_files: List[str] = field(
        default=None, metadata={"help": "Path to test json files"}
    )

@dataclass
class TrainingArgsConfig(transformers.TrainingArguments):
    fp16 = True
    gradient_accumulation_steps = 2,
    optim = 'paged_adamw_32bit',
    # learning_rate=learning_rate,
    max_grad_norm = 0.3,
    # max_steps: int = ,
    warmup_ratio = 0.03,
    group_by_length = True
    lr_scheduler_type = 'constant',
    gradient_checkpointing = False,
    # neftune_noise_alpha: int = ,

    seed: int = 42
    output_dir: str = "./runs/exp_3"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    auto_find_batch_size: bool = True
    logging_dir: str = "./runs/exp_3/logs"
    logging_steps: int = 10
    run_name: str = 'exp_3'
    load_best_model_at_end: bool = True
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2


class PairsDataset(Dataset):
    def __init__(self, x, y, tokenizer_name, max_length=510):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.x = self.tokenizer(x, padding='max_length', max_length=max_length, is_split_into_words=True),
        self.y = self.tokenizer(y, padding='max_length', max_length=max_length, is_split_into_words=True),

    def __getitem__(self, idx):
        assert idx <= len(self.x['input_ids']), (idx, len(self.x['input_ids']))
        item = {key: val[idx] for key, val in self.x.items()}

        item['labels'] = self.y['input_ids'][idx]
        if IS_ENCODER_DECODER: item['decoder_attention_mask'] = self.y['attention_mask'][idx]

        return item

    @property
    def n(self):
        return len(self.x['input_ids'])

    def __len__(self):
        return self.n


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


def generate_inoutput(df):
    '''
    ### Instruction: find first 10 words where AI generated text starts .

    ### Input: _input_

    ### Response:
    '''
    pattern = "### Instruction: find first 10 words where AI generated text starts .\n\n### Input:".split(' ')
    end_pattern = "\n\n### Response:".split(' ')
    input = []
    output = []

    for _, text, label in df.values:
        words = text.split()[label:label+10]
        # print('\n', words)
        output.append(words)
        input.append(pattern + text.split() + end_pattern)

    return input, output


def evaluate_position_difference(actual_position, predicted_position):
    """
    Compute the absolute difference between the actual and predicted start positions.

    Args:
    - actual_position (int): Actual start position of machine-generated text.
    - predicted_position (int): Predicted start position of machine-generated text.

    Returns:
    - int: Absolute difference between the start positions.
    """
    return abs(actual_position - predicted_position)


def get_start_position(sequence, mapping=None, token_level=True):
    """
    Get the start position from a sequence of labels or predictions.

    Args:
    - sequence (np.array): A sequence of labels or predictions.
    - mapping (np.array): Mapping from index to word for the sequence.
    - token_level (bool): If True, return positional indices; else, return word mappings.

    Returns:
    - int or str: Start position in the sequence.
    """
    # Locate the position of label '1'

    if mapping is not None:
        mask = mapping != -100
        sequence = sequence[mask]
        mapping = mapping[mask]

    index = np.where(sequence == 1)[0]
    value = index[0] if index.size else (len(sequence) - 1)

    if not token_level:
        value = mapping[value]

    return value


def evaluate_machine_start_position(
    labels, predictions, idx2word=None, token_level=False
):
    """
    Evaluate the starting position of machine-generated text in both predicted and actual sequences.

    Args:
    - labels (np.array): Actual labels.
    - predictions (np.array): Predicted labels.
    - idx2word (np.array): Mapping from index to word for each sequence in the batch.
    - token_level (bool): Flag to determine if evaluation is at token level. If True, return positional indices; else, return word mappings.

    Returns:
    - float: Mean absolute difference between the start positions in predictions and actual labels.
    """

    predicted_positions = predictions.argmax(axis=-1)

    actual_starts = []
    predicted_starts = []

    if not token_level and idx2word is None:
        raise ValueError(
            "idx2word must be provided if evaluation is at word level (token_level=False)"
        )

    for idx in range(labels.shape[0]):
        # Remove padding
        mask = labels[idx] != -100
        predict, label, mapping = (
            predicted_positions[idx][mask],
            labels[idx][mask],
            idx2word[idx][mask] if not token_level else None,
        )

        # If token_level is True, just use the index; otherwise, map to word
        predicted_value = get_start_position(predict, mapping, token_level)
        actual_value = get_start_position(label, mapping, token_level)

        predicted_starts.append(predicted_value)
        actual_starts.append(actual_value)

    position_differences = [
        evaluate_position_difference(actual, predict)
        for actual, predict in zip(actual_starts, predicted_starts)
    ]
    mean_position_difference = np.mean(position_differences)

    return mean_position_difference


def compute_metrics(p):
    pred, labels = p
    mean_absolute_diff = evaluate_machine_start_position(labels, pred, token_level=True)

    return {
        "mean_absolute_diff": mean_absolute_diff,
    }




if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = 'nlp_hw2'

    parser = transformers.HfArgumentParser(
        (ModelConfig, DatasetConfig, TrainingArgsConfig)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("Model Arguments: ", model_args)
    print("Data Arguments: ", data_args)
    print("Training Arguments: ", training_args)
    dir = training_args.output_dir.split('/')[-1]
    training_args.run_name = f'{dir}_{training_args.num_train_epochs}ep_{training_args.per_device_train_batch_size}b'
    # Set seed
    transformers.set_seed(training_args.seed)

    model_path = model_args.model_path

    df = get_dataset(data_args.train_file)
    df_dev = get_dataset(data_args.dev_file)

    input, output = generate_inoutput(df)
    input_dev, output_dev = generate_inoutput(df_dev)

    print('Loaded dataset!')

    print('---------------------')
    print(' '.join(input[-1]))
    print(' '.join(output[-1]))
    print('---------------------')

    if (
        training_args.do_eval or training_args.do_predict
    ) and not training_args.do_train:
        output_dir = training_args.output_dir
        if not os.path.exists(output_dir):
            raise ValueError(
                f"Output directory ({output_dir}) does not exist. Please train the model first."
            )

        # Find the best model checkpoint
        ckpt_paths = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda x: int(x.split("-")[-1]),
        )

        if not ckpt_paths:
            raise ValueError(
                f"Output directory ({output_dir}) does not contain any checkpoint. Please train the model first."
            )

        state = TrainerState.load_from_json(
            os.path.join(ckpt_paths[-1], "trainer_state.json")
        )
        best_model_path = state.best_model_checkpoint or model_args.model_path
        if state.best_model_checkpoint is None:
            logger.info(
                "No best model checkpoint found. Using the default model checkpoint."
            )
        print(f"Best model path: {best_model_path}")
        model_path = best_model_path

    # model = AutoModelForTokenClassification.from_pretrained(model_path, config=model_args)

    # Bits and Bytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    
    
    model = AutoModelForCausalLM.from_pretrained(model_path, config=model_args)

    # IS_ENCODER_DECODER = True
    IS_ENCODER_DECODER = False
    MAX_LENGTH = 510

    train_set = PairsDataset(input, output, model_path)
    dev_set = PairsDataset(input_dev, output_dev, model_path)

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64


    lora_target_modules = [
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj",
    ]

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = lora_target_modules
    )


    max_seq_length = 128 # 510


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=dev_set,
        peft_config=peft_config,
        dataset_text_field='prompt',
        max_seq_length=max_seq_length,
        tokenizer=train_set.tokenizer,
        args=training_args,
    )

    # trainer = transformers.Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_set,
    #     eval_dataset=dev_set,
    #     tokenizer=train_set.tokenizer,
    #     # compute_metrics=compute_metrics,
    # )

    if training_args.do_train:
        logger.info("Training...")
        logger.info("*** Train Dataset ***")
        logger.info(f"Number of samples: {len(train_set)}")
        logger.info("*** Dev Dataset ***")
        logger.info(f"Number of samples: {len(dev_set)}")

        trainer.train()

        logger.info("Training completed!")

    if training_args.do_eval:
        logger.info("Evaluating...")
        logger.info("*** Dev Dataset ***")
        logger.info(f"Number of samples: {len(dev_set)}")

        metrics = trainer.evaluate()
        logger.info(f"Metrics: {metrics}")
        trainer.save_metrics("eval", metrics)

        logger.info("Evaluation completed!")

    if training_args.do_predict:

        test_sets = [get_dataset(data_args.dev_file)]

        logger.info("Predicting...")
        logger.info("*** Test Datasets ***")
        logger.info(f"Number of samples: {len(test_sets)}")

        for idx, test_set in enumerate(test_sets):
            logger.info(f"Test Dataset {idx + 1}")
            logger.info(f"Number of samples: {len(test_set)}")

            predictions, _, _ = trainer.predict(test_set)
            logger.info("Predictions completed!")

            df = pd.DataFrame(
                {
                    "id": [i["id"] for i in test_set],
                    "label": [
                        get_start_position(
                            i[0],
                            np.array(i[1]["corresponding_word"]),
                            token_level=False,
                        )
                        for i in list(zip(predictions.argmax(axis=-1), test_set))
                    ],
                }
            )
            import os

            file_name = os.path.basename(data_args.test_files[idx])
            file_dirs = os.path.join(training_args.output_dir, "predictions")
            os.makedirs(file_dirs, exist_ok=True)
            file_path = os.path.join(file_dirs, file_name)
            records = df.to_dict("records")
            with open(file_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")










    # print('Ready to train!')

    # trainer.train()

    # # saved_name = '_'.join([model_name.split('-')[0], str(arg_parser.num_epochs)+'ep',
    # saved_name = '_'.join([model_name.split('/')[0], str(arg_parser.num_epochs)+'ep',
    #                        str(arg_parser.train_batch_size)+'_'+str(arg_parser.val_batch_size)+'b'])
    # dir = arg_parser.output_dir + '/' + saved_name

    # print(dir)
    # # torch.save(model.state_dict(), saved_name+'.pth')
    # # tokenizer.save_pretrained(dir+"_tok")

    # # print('Trained model is saved!')
    # print('Model finished!')

    # wandb.finish()

    # print('\nEvaluation:\n')


    # df = get_dataset(arg_parser.test_path)
    # labels, words = generate_labels(df)
    # df['labels'] = labels
    # df['words'] = words

    # test_dataset = align_labels(df['words'].to_list(), df['labels'].to_list())
    # # pred = trainer.predict(test_dataset)

    # predictions, labels, _ = trainer.predict(PairsDataset(test_dataset))
    # predictions = np.argmax(predictions, axis=2)

    # true_predictions = [
    #     [p for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    # true_labels = [
    #     [l for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]

    # metric = load_metric("seqeval")
    # results = metric.compute(predictions=true_predictions, references=true_labels)
    
    # print(f'Score for predictions with {model_name}:')
    # print(results)


    # df_post = pd.DataFrame({"id": [id for id in df.id], "label": get_label(true_predictions)})
    
    # file_name = os.path.basename('pred.jsonl')
    # file_dirs = os.path.join(arg_parser.output_dir, saved_name)
    # os.makedirs(file_dirs, exist_ok=True)
    # file_path = os.path.join(file_dirs, file_name)
    
    # print(f'Saving into the file {file_path}')

    # records = df_post.to_dict("records")
    # with open(file_path, "w") as f:
    #     for record in records:
    #         f.write(json.dumps(record) + "\n")

    # print('A miracle happened ^-^/***')



































# if __name__ == "__main__":
#     args = get_args()

#     for arg in vars(args):
#         print(arg, getattr(args, arg))



#     model_name = args.model_name

#     ## Bits and Bytes config
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         # bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True
#     )


#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         trust_remote_code=True,
#         use_flash_attention_2=args.use_flash_attention_2,
#     )


#     ## Enable gradient checkpointing
#     if args.gradient_checkpointing:
#         model.gradient_checkpointing_enable()

#     ## Prepare model for k-bit training
#     # model = prepare_model_for_kbit_training(model)


#     ## Print the number of trainable parameters
#     print_trainable_parameters(model)

#     ## Silence the warnings
#     model.config.use_cache = False

#     ## Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = 'right'


#     output_dir = args.output_dir

#     per_device_train_batch_size =  args.per_device_train_batch_size
#     per_device_val_batch_size = args.per_device_val_batch_size
#     gradient_accumulation_steps = args.gradient_accumulation_steps


#     epoch_steps = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)


#     optim = args.optim

#     save_steps = args.save_steps
#     logging_steps = args.logging_steps
#     learning_rate = args.learning_rate
#     max_grad_norm = args.max_grad_norm



#     print(f"save_steps: {save_steps}")
#     print(f"logging_steps: {logging_steps}")


#     max_steps = epoch_steps * 10

#     warmup_ratio = args.warmup_ratio
#     lr_scheduler_type = args.lr_scheduler_type


#     output_dir = args.output_dir + f"/{model_name.split('/')[-1]}"
#     loggig_dir = args.logging_dir + f"/{model_name.split('/')[-1]}" + f"/logs"
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     Path(loggig_dir).mkdir(parents=True, exist_ok=True)
#     print(f"Saving the model to {output_dir}")


#     training_arguments = TrainingArguments(
#         output_dir=output_dir,
#         per_device_train_batch_size=per_device_train_batch_size,
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         optim=optim,
#         per_device_eval_batch_size=per_device_val_batch_size,
#         evaluation_strategy=args.evaluation_strategy,
#         do_train=args.do_train,
#         do_eval=args.do_eval,
#         # eval_steps=10,
#         eval_steps=args.eval_steps,
#         run_name=args.run_name,
#         save_steps=save_steps,
#         logging_steps=logging_steps,
#         learning_rate=learning_rate,
#         fp16=True,
#         max_grad_norm=max_grad_norm,
#         max_steps=max_steps,
#         warmup_ratio=warmup_ratio,
#         group_by_length=True,
#         lr_scheduler_type=lr_scheduler_type,
#         report_to=args.report_to,
#         gradient_checkpointing=args.gradient_checkpointing,
#         neftune_noise_alpha=0.1,
#         logging_dir=loggig_dir,
#     )


#     lora_alpha = args.lora_alpha
#     lora_dropout = args.lora_dropout
#     lora_r = args.lora_r


#     lora_target_modules = args.lora_target_modules
#     # [
#     #     "q_proj",
#     #     "up_proj",
#     #     "o_proj",
#     #     "k_proj",
#     #     "down_proj",
#     #     "gate_proj",
#     #     "v_proj",
#     # ]

#     peft_config = LoraConfig(
#         lora_alpha=lora_alpha,
#         lora_dropout=lora_dropout,
#         r=lora_r,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules = lora_target_modules
#     )


#     max_seq_length = args.max_seq_length


#     trainer = SFTTrainer(
#         model=model,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         peft_config=peft_config,
#         dataset_text_field=args.field,
#         max_seq_length=max_seq_length,
#         tokenizer=tokenizer,
#         args=training_arguments,
#     )


#     for name, module in trainer.model.named_modules():
#         if "norm" in name:
#             module = module.to(torch.float32)


#     if args.checkpoint_path:
#         trainer.train(resume_from_checkpoint=args.checkpoint_path)
#     else:
#         trainer.train()


#     # trainer.save_model()


#     print("Done training")
#     print(trainer.model)










































# import torch
# import json
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers.trainer_callback import TrainerState
# import transformers
# import pandas as pd
# import numpy as np

# from dataclasses import dataclass, field
# from typing import List#, Any, Optional
# import logging
# import glob
# import os

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()


# @dataclass
# class ModelConfig:
#     model_path: str = "allenai/longformer-base-4096"
#     trust_remote_code: bool = True,
#     ignore_mismatched_sizes: bool = True, #dslim
#     # num_labels: int = 2 # YES!!!
#     # num_hidden_layers: int = 26 # 12 def
#     # # classifier_dropout: float = 0.1


# @dataclass
# class DatasetConfig:
#     train_file: str = field(default=None, metadata={"help": "Path to train jsonl file"})
#     dev_file: str = field(default=None, metadata={"help": "Path to dev jsonl file"})
#     test_files: List[str] = field(
#         default=None, metadata={"help": "Path to test json files"}
#     )


# @dataclass
# class TrainingArgsConfig(transformers.TrainingArguments):
#     seed: int = 42
#     output_dir: str = "./runs/exp_3"
#     num_train_epochs: int = 10
#     per_device_train_batch_size: int = 32
#     per_device_eval_batch_size: int = 32
#     auto_find_batch_size: bool = True
#     logging_dir: str = "./runs/exp_3/logs"
#     logging_steps: int = 10
#     run_name: str = 'exp_3'
#     load_best_model_at_end: bool = True
#     evaluation_strategy: str = "epoch"
#     save_strategy: str = "epoch"
#     save_total_limit: int = 2


# class Semeval_Data(torch.utils.data.Dataset):
#     # def __init__(self, data_path, tokenizer_name, max_length=1024, inference=False, debug=False):
#     def __init__(self, data_path, tokenizer_name, max_length=510, inference=False, debug=False): # BERT
#         with open(data_path, "r") as f:
#             self.data = [json.loads(line) for line in f]
#         self.inference = inference
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.max_length = max_length
#         self.debug = debug

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         text = self.data[idx]["text"]
#         id = self.data[idx]["id"]
#         label = None
#         labels_available = "label" in self.data[idx]

#         if labels_available:
#             label = self.data[idx]["label"]

#         if self.debug and not self.inference:
#             print("Orignal Human Position: ", label)

#         labels = []
#         corresponding_word = []
#         tokens = []
#         input_ids = []
#         attention_mask = []

#         for jdx, word in enumerate(text.split(" ")):
#             word_encoded = self.tokenizer.tokenize(word)
#             sub_words = len(word_encoded)

#             if labels_available:
#                 is_machine_text = 1 if jdx >= label else 0
#                 labels.extend([is_machine_text] * sub_words)

#             corresponding_word.extend([jdx] * sub_words)
#             tokens.extend(word_encoded)
#             input_ids.extend(self.tokenizer.convert_tokens_to_ids(word_encoded))
#             attention_mask.extend([1] * sub_words)

#         ###Add padding to labels as -100
#         if len(input_ids) < self.max_length - 2:
#             input_ids = (
#                 [0] + input_ids + [2] + [1] * (self.max_length - len(input_ids) - 2)
#             )
#             if labels_available:
#                 labels = [-100] + labels + [-100] * (self.max_length - len(labels) - 1)

#             attention_mask = (
#                 [1]
#                 + attention_mask
#                 + [1]
#                 + [0] * (self.max_length - len(attention_mask) - 2)
#             )
#             corresponding_word = (
#                 [-100]
#                 + corresponding_word
#                 + [-100] * (self.max_length - len(corresponding_word) - 1)
#             )
#             tokens = (
#                 ["<s>"]
#                 + tokens
#                 + ["</s>"]
#                 + ["<pad>"] * (self.max_length - len(tokens) - 2)
#             )
#         else:
#             # Add -100 for CLS and SEP tokens
#             input_ids = [0] + input_ids[: self.max_length - 2] + [2]

#             if labels_available:
#                 labels = [-100] + labels[: self.max_length - 2] + [-100]

#             corresponding_word = (
#                 [-100] + corresponding_word[: self.max_length - 2] + [-100]
#             )
#             attention_mask = [1] + attention_mask[: self.max_length - 2] + [1]
#             tokens = ["<s>"] + tokens[: self.max_length - 2] + ["</s>"]

#         encoded = {}
#         if labels_available:
#             encoded["labels"] = torch.tensor(labels)

#         encoded["input_ids"] = torch.tensor(input_ids)
#         encoded["attention_mask"] = torch.tensor(attention_mask)

#         if labels_available:
#             if encoded["input_ids"].shape != encoded["labels"].shape:
#                 print("Input IDs Shape: ", encoded["input_ids"].shape)
#                 print("Labels Shape: ", encoded["labels"].shape)
#             assert encoded["input_ids"].shape == encoded["labels"].shape

#         if self.debug and not self.inference:
#             print("Tokenized Human Position: ", labels.index(1))
#             print("Original Human Position: ", label)
#             print("Full Human Text:", text)
#             print("\n")
#             print("Human Text Truncated:", text.split(" ")[:label])
#             print("\n")
#             encoded["partial_human_review"] = " ".join(text.split(" ")[:label])

#         if self.inference:
#             encoded["text"] = text
#             encoded["id"] = id
#             encoded["corresponding_word"] = corresponding_word

#         return encoded

