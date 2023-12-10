#MODEL_NAME='hiendang7613/xlmr-lstm-crf-resume-ner'
#DIR='xlmr-lstm-crf_7ep'

# MODEL_NAME='xlm-roberta-base' # 'xlm-roberta-base' # xlm-roberta-large-finetuned-conll03-english / xlm-roberta-large # 510
# DIR='xlm-roberta_3ep_18hide_075weight'

MODEL_NAME='microsoft/mdeberta-v3-base'
DIR='mdeberta_3ep_18hide'

#MODEL_NAME='symanto/xlm-roberta-base-snli-mnli-anli-xnli'
#DIR='symanto_5ep_18hide'

#MODEL_NAME='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
#DIR='moritz_3ep_18hide'

# MODEL_NAME='microsoft/deberta-v3-base'
# DIR='deberta_3ep'

# MODEL_NAME='roberta-base' # 'roberta-base' / roberta-large # 510
# DIR='roberta_5ep'


echo " starting "
echo $MODEL_NAME
echo $DIR

seed_value=42

mkdir -p experiments/$DIR
> "experiments/$DIR/error.txt"
> "experiments/$DIR/output.txt"

python taska.py \
  --model_path $MODEL_NAME \
  --train_file "data/subtaskA_train_monolingual.jsonl" \
  --test_file "data/subtaskA_dev_monolingual.jsonl" \
  --do_train True \
  --do_predict True \
  --seed $seed_value \
  --output_dir "experiments/$DIR" \
  --logging_dir "experiments/$DIR/logs" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 20 \
  --per_device_eval_batch_size 20 \
  --auto_find_batch_size False \
  --logging_steps 100 \
  --load_best_model_at_end True \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 3 \
  --weight_decay 0.001 \
  --learning_rate 2e-5 \
  2> "experiments/$DIR/error.txt" > "experiments/$DIR/output.txt"

echo ".............. ending .............."


python scorer.py \
  --gold_file_path "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/data/subtaskA_dev_monolingual.jsonl" \
  --pred_file_path "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/predictions/subtask_a_monolingual.jsonl" \
  2> "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/result.txt" \

