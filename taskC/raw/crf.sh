# MODEL_NAME='allenai/longformer-base-4096' # baseline
# DIR='allenai_10ep_6hide'

# MODEL_NAME='roberta-base' # 'roberta-base' / roberta-large # 15 ep 510
# DIR='roberta_5ep'
# MODEL_NAME='dslim/bert-base-NER' # 20ep ignore_mismatched_sizes 510
# DIR='dslim_f'

MODEL_NAME='xlm-roberta-base' # 'xlm-roberta-base' # xlm-roberta-large-finetuned-conll03-english / xlm-roberta-large
# DIR='xlm-roberta_2l_20ep_1drop' # 510
DIR='crf_xlm-roberta_20ep_2l_18hide' # 510

# MODEL_NAME='hiendang7613/xlmr-lstm-crf-resume-ner'
# DIR='xlmr-lstm-crf_2l_18hide'
# DIR='xlmr-lstm-crf'


echo " starting "
echo $MODEL_NAME
echo $DIR

seed_value=42

mkdir -p experiments/$DIR
> "experiments/$DIR/error.txt"
> "experiments/$DIR/output.txt"

python crf.py \
  --model_path $MODEL_NAME \
  --train_file "data/subtaskC_train.jsonl" \
  --load_best_model_at_end True \
  --dev_file "data/subtaskC_dev.jsonl" \
  --test_files data/subtaskC_dev.jsonl \
  --metric_for_best_model "eval_mean_absolute_diff" \
  --do_train True \
  --do_predict True \
  --seed $seed_value \
  --output_dir "experiments/$DIR" \
  --logging_dir "experiments/$DIR/logs" \
  --num_train_epochs 20 \
  --per_device_train_batch_size 26 \
  --per_device_eval_batch_size 26 \
  --auto_find_batch_size False \
  --logging_steps 10 \
  --load_best_model_at_end True \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  2> "experiments/$DIR/error.txt" > "experiments/$DIR/output.txt"

echo ".............. ending .............."


python scorer.py \
  --gold_file_path "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/data/subtaskC_dev.jsonl" \
  --pred_file_path "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/predictions/subtaskC_dev.jsonl" \
  2> "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/result.txt" \
#   > "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/logs.txt"

