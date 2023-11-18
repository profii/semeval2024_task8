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

# BERT, RoBERTa, mBERT, XLM-R
MODEL_NAME="bert-base-uncased"

python my.py \
  --model_name $MODEL_NAME \
  --output_dir "experiments/$MODEL_NAME" \
  --logging_dir "experiments/$MODEL_NAME/logs" \
  --num_epochs 4 \
  --save_total_limit 2 \
  --train_batch_size 4 \
  --val_batch_size 4 \
  --save_steps 100 \
  --logging_steps 10 \


echo " ending "

