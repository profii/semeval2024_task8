# MODEL_NAME='allenai/longformer-base-4096' # baseline
# DIR='allenai_10ep_6hide'

# MODEL_NAME='roberta-base' # 'roberta-base' / roberta-large # 15 ep 510
# DIR='roberta_5ep'
# MODEL_NAME='dslim/bert-base-NER' # 20ep ignore_mismatched_sizes 510
# DIR='dslim_f'

# MODEL_NAME='xlm-roberta-base' # 'xlm-roberta-base' # xlm-roberta-large-finetuned-conll03-english / xlm-roberta-large
# DIR='xlm-roberta_2l_20ep_1drop' # 510
# DIR='xlm-roberta_2l_26hide' # 510

# MODEL_NAME='hiendang7613/xlmr-lstm-crf-resume-ner'
# DIR='xlmr-lstm-crf_2l_18hide'
# DIR='xlmr-lstm-crf'
# DIR='xlmr-lstm-crf_2l_20ep'


# MODEL_NAME='xlm-roberta-large' # xlm-roberta-large-finetuned-conll03-english / xlm-roberta-large
# DIR='xlm-large_f' # 510
# DIR='xlm-fine_f' # 510
# MODEL_NAME='sentence-transformers/all-MiniLM-L6-v2' # 510
# DIR='minilm_20ep' #overfit

# MODEL_NAME='numind/generic-entity_recognition_NER-v1' #15ep 510
# DIR='numind'

MODEL_NAME='microsoft/mdeberta-v3-base'
DIR='mdeberta_20ep'
# DIR='deberta_5ep'

# MODEL_NAME='alexeyak/deberta-v3-base-ner-B'
# DIR='deberta_alexeyak'

# MODEL_NAME=''
# DIR='deberta'
# MODEL_NAME='microsoft/deberta-v3-base' #15ep 510
# DIR='deberta'



#NO:
# MODEL_NAME="bert-base-uncased" # bert-large-uncased add_prefix_space=False
# DIR='bert'
# MODEL_NAME='bert-base-multilingual-cased'
# DIR='bert-multilingual'
# MODEL_NAME='jinaai/jina-embeddings-v2-base-en'
# DIR='jina'


# MODEL_NAME='mistralai/Mistral-7B-v0.1'
# DIR='mistral'

echo " starting "
echo $MODEL_NAME
echo $DIR

seed_value=42

mkdir -p experiments_c/$DIR
> "experiments_c/$DIR/error.txt"
> "experiments_c/$DIR/output.txt"

python base.py \
  --model_path $MODEL_NAME \
  --train_file "../data/subtaskC_train.jsonl" \
  --load_best_model_at_end True \
  --dev_file "../data/subtaskC_dev.jsonl" \
  --test_files ../data/subtaskC_dev.jsonl \
  --metric_for_best_model "eval_mean_absolute_diff" \
  --do_train True \
  --do_predict True \
  --seed $seed_value \
  --output_dir "experiments_c/$DIR" \
  --logging_dir "experiments_c/$DIR/logs" \
  --num_train_epochs 20 \
  --per_device_train_batch_size 18 \
  --per_device_eval_batch_size 18 \
  --auto_find_batch_size False \
  --logging_steps 10 \
  --load_best_model_at_end True \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 4 \
  2> "experiments_c/$DIR/error.txt" > "experiments_c/$DIR/output.txt"

echo ".............. ending .............."


python scorer.py \
  --gold_file_path "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/data/subtaskC_dev.jsonl" \
  --pred_file_path "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/TaskC/experiments_c/$DIR/predictions/subtaskC_dev.jsonl" \
  2> "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/TaskC/experiments_c/$DIR/result.txt" \
#   > "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/logs.txt"

