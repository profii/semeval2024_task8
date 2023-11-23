#!/bin/bash
#SBATCH --job-name=my # Job name
#SBATCH --error=/home/anastasiia.demidova/nlp_hw2/experiments/logs/%j%x.err # error file
#SBATCH --output=/home/anastasiia.demidova/nlp_hw2/experiments/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l1-005


WANDB_PROJECT=nlp_hw2
echo $WANDB_PROJECT

MODEL_NAME = 'allenai/longformer-base-4096' # baseline

# MODEL_NAME="bert-base-uncased" # bert-large-uncased !!!!!!add_prefix_space=False!!!!!
# MODEL_NAME = 'roberta-base' # roberta-large
# MODEL_NAME = 'xlm-roberta-base' # xlm-roberta-large-finetuned-conll03-english / xlm-roberta-large
# MODEL_NAME = 'bert-base-multilingual-cased'

# MODEL_NAME = 'numind/generic-entity_recognition_NER-v1'
# MODEL_NAME = 'dslim/bert-base-NER'
# MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# MODEL_NAME = 'jinaai/jina-embeddings-v2-base-en'



python my.py \
  --wandb $WANDB_PROJECT \
  --model_name $MODEL_NAME \
  --output_dir "experiments/$MODEL_NAME" \
  --logging_dir "experiments/$MODEL_NAME/logs" \
  --num_epochs 4 \
  --save_total_limit 2 \
  --train_batch_size 4 \
  --val_batch_size 4 \
  --save_steps 100 \
  --logging_steps 10 \
  # --add_prefix_space False \

echo " ending "

