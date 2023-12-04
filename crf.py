import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.trainer_callback import TrainerState
import transformers
import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from typing import List#, Any, Optional
import logging
import glob
import os

from torchcrf import CRF
# from TorchCRF import CRF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@dataclass
class ModelConfig:
    model_path: str = "allenai/longformer-base-4096"
    trust_remote_code: bool = True,
    ignore_mismatched_sizes: bool = True, #dslim
    num_labels: int = 2 # YES!!!
    num_hidden_layers: int = 18 # 12 def
    # classifier_dropout: float = 0.1

    # hidden_act: str = "relu" # def:"gelu"; "relu", "silu" and "gelu_new"
    # num_attention_heads: int = 10 # 12 def
    # position_embedding_type: str = "relative_key" # def:"absolute"; "relative_key", "relative_key_query"
    # hidden_dropout_prob: float = 0.3
    # attention_probs_dropout_prob: float = 0.25


@dataclass
class DatasetConfig:
    train_file: str = field(default=None, metadata={"help": "Path to train jsonl file"})
    dev_file: str = field(default=None, metadata={"help": "Path to dev jsonl file"})
    test_files: List[str] = field(
        default=None, metadata={"help": "Path to test json files"}
    )


@dataclass
class TrainingArgsConfig(transformers.TrainingArguments):
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


class Semeval_Data(torch.utils.data.Dataset):
    # def __init__(self, data_path, tokenizer_name, max_length=1024, inference=False, debug=False):
    def __init__(self, data_path, tokenizer_name, max_length=510, inference=False, debug=False): # BERT
        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.inference = inference
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.debug = debug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        id = self.data[idx]["id"]
        label = None
        labels_available = "label" in self.data[idx]

        if labels_available:
            label = self.data[idx]["label"]

        if self.debug and not self.inference:
            print("Orignal Human Position: ", label)

        labels = []
        corresponding_word = []
        tokens = []
        input_ids = []
        attention_mask = []

        for jdx, word in enumerate(text.split(" ")):
            word_encoded = self.tokenizer.tokenize(word)
            sub_words = len(word_encoded)

            if labels_available:
                is_machine_text = 1 if jdx >= label else 0
                labels.extend([is_machine_text] * sub_words)

            corresponding_word.extend([jdx] * sub_words)
            tokens.extend(word_encoded)
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(word_encoded))
            attention_mask.extend([1] * sub_words)

        ###Add padding to labels as -100
        if len(input_ids) < self.max_length - 2:
            input_ids = (
                [0] + input_ids + [2] + [1] * (self.max_length - len(input_ids) - 2)
            )
            if labels_available:
                labels = [-100] + labels + [-100] * (self.max_length - len(labels) - 1)

            attention_mask = (
                [1]
                + attention_mask
                + [1]
                + [0] * (self.max_length - len(attention_mask) - 2)
            )
            corresponding_word = (
                [-100]
                + corresponding_word
                + [-100] * (self.max_length - len(corresponding_word) - 1)
            )
            tokens = (
                ["<s>"]
                + tokens
                + ["</s>"]
                + ["<pad>"] * (self.max_length - len(tokens) - 2)
            )
        else:
            # Add -100 for CLS and SEP tokens
            input_ids = [0] + input_ids[: self.max_length - 2] + [2]

            if labels_available:
                labels = [-100] + labels[: self.max_length - 2] + [-100]

            corresponding_word = (
                [-100] + corresponding_word[: self.max_length - 2] + [-100]
            )
            attention_mask = [1] + attention_mask[: self.max_length - 2] + [1]
            tokens = ["<s>"] + tokens[: self.max_length - 2] + ["</s>"]

        encoded = {}
        if labels_available:
            encoded["labels"] = torch.tensor(labels)

        encoded["input_ids"] = torch.tensor(input_ids)
        encoded["attention_mask"] = torch.tensor(attention_mask)

        if labels_available:
            if encoded["input_ids"].shape != encoded["labels"].shape:
                print("Input IDs Shape: ", encoded["input_ids"].shape)
                print("Labels Shape: ", encoded["labels"].shape)
            assert encoded["input_ids"].shape == encoded["labels"].shape

        if self.debug and not self.inference:
            print("Tokenized Human Position: ", labels.index(1))
            print("Original Human Position: ", label)
            print("Full Human Text:", text)
            print("\n")
            print("Human Text Truncated:", text.split(" ")[:label])
            print("\n")
            encoded["partial_human_review"] = " ".join(text.split(" ")[:label])

        if self.inference:
            encoded["text"] = text
            encoded["id"] = id
            encoded["corresponding_word"] = corresponding_word

        return encoded


class CustomModelForTokenClassification(AutoModelForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits
        loss = None

        if labels is not None:
            # loss = -self.crf(logits, labels, mask=attention_mask.byte())
            loss = -self.crf(torch.log_softmax(logits, 2), labels, mask=attention_mask.byte(), reduction='mean')
            outputs = (loss,) + outputs

        return outputs

        # outputs = self.bert(input_ids, attention_mask=attention_mask)
        # sequence_output = torch.stack((outputs[1][-1], outputs[1][-2], outputs[1][-3], outputs[1][-4])).mean(dim=0)
        # sequence_output = self.dropout(sequence_output)
        # emission = self.classifier(sequence_output) # [32,256,17]
        # if labels is not None:
        #     labels=labels.reshape(attention_mask.size()[0],attention_mask.size()[1])
        #     loss = -self.crf(log_soft(emission, 2), labels, mask=attention_mask.type(torch.uint8), reduction='mean')
        #     prediction = self.crf.decode(emission, mask=attention_mask.type(torch.uint8))
        #     return [loss, prediction]
        

# class CustomModel(nn.Module):
#   def __init__(self,checkpoint,num_labels): 
#     super(CustomModel,self).__init__() 
#     self.num_labels = num_labels 

#     #Load Model with given checkpoint and extract its body
#     self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
#     self.dropout = nn.Dropout(0.1) 
#     self.classifier = nn.Linear(768,num_labels) # load and initialize weights

#   def forward(self, input_ids=None, attention_mask=None,labels=None):
#     #Extract outputs from the body
#     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

#     #Add custom layers
#     sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

#     logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
#     loss = None
#     if labels is not None:
#       loss_fct = nn.CrossEntropyLoss()
#       loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
#     return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)


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

    # 4. Load model
    model = CustomModelForTokenClassification.from_pretrained(model_path, config=model_args)

    # model = AutoModelForTokenClassification.from_pretrained(model_path, config=model_args)
    # model = AutoModelForTokenClassification.from_pretrained(
    #     model_path, num_labels=2, trust_remote_code=True, ignore_mismatched_sizes=True, #dslim
    #     hidden_dropout_prob = 0.3, attention_probs_dropout_prob = 0.25 # dropout
    # )

    train_set = Semeval_Data(data_args.train_file, model_args.model_path)
    dev_set = Semeval_Data(data_args.dev_file, model_args.model_path)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=train_set.tokenizer,
        compute_metrics=compute_metrics,
    )

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
        test_sets = []
        for test_file in data_args.test_files:
            test_set = Semeval_Data(test_file, model_args.model_path, inference=True)
            test_sets.append(test_set)
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
