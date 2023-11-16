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

SEED=42

python my.py \
  --model_name=$MODEL_NAME \
  --train_path="/home/anastasiia.demidova/nlp_hw2/subtaskC/data/subtaskC_train.jsonl" \
  --test_path="/home/anastasiia.demidova/nlp_hw2/subtaskC/data/subtaskC_dev.jsonl" \
  --seed=$SEED \
  --logging_dir "/home/anastasiia.demidova/nlp_hw2/semeval2024/semeval2024_task8/experiments/$MODEL_NAME/logs" \




  # --load_best_model_at_end True \
  # --metric_for_best_model "eval_mean_absolute_diff" \
  # --greater_is_better False \
  # --do_train True \
  # --do_predict True \
  # --seed $seed_value \
  # --output_dir "./runs/$exp_name" \
  # --logging_dir "./runs/$exp_name/logs" \
  # --num_train_epochs 10 \
  # --per_device_train_batch_size 32 \
  # --per_device_eval_batch_size 32 \
  # --auto_find_batch_size True \
  # --logging_steps 10 \
  # --load_best_model_at_end True \
  # --evaluation_strategy "epoch" \
  # --save_strategy "epoch" \
  # --save_total_limit 2



# --save_steps=1000 \
# --eval_steps=10000 \
# --do_eval=1 \
# --report_to="all" \
# --logging_steps=500 \
# --logging_dir="experiments/$MODEL_NAME" \
# --model_name=$MODEL_NAME \
# --run_name=$MODEL_NAME \
# --per_device_train_batch_size=4 \
# --per_device_val_batch_size=4 \
# --model="english" \
# --gradient_accumulation_steps=2 \
# --gradient_checkpointing=1 \
# --checkpoint_path="experiments/Llama-2-7b-hf/checkpoint-39000"



echo " ending "

