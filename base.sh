# WANDB_PROJECT=nlp_hw2
# echo $WANDB_PROJECT


MODEL_NAME='allenai/longformer-base-4096' # baseline
DIR='allenai'

# MODEL_NAME='roberta-base' # roberta-large # 15 ep 510
# DIR='roberta'
# MODEL_NAME='xlm-roberta-base' # xlm-roberta-large-finetuned-conll03-english / xlm-roberta-large
# DIR='xlm' # 20 510

# MODEL_NAME='numind/generic-entity_recognition_NER-v1' #15ep 510
# DIR='numind'

# MODEL_NAME='dslim/bert-base-NER' # 20ep ignore_mismatched_sizes 510
# DIR='dslim'
# MODEL_NAME='sentence-transformers/all-MiniLM-L6-v2' # 510
# DIR='minilm'
# # MODEL_NAME='jinaai/jina-embeddings-v2-base-en'
# DIR='jina'
#NO:
# MODEL_NAME="bert-base-uncased" # bert-large-uncased !!!!!!add_prefix_space=False!!!!!
# DIR='bert'
# MODEL_NAME='bert-base-multilingual-cased'
# DIR='bert-multilingual'


echo " starting "
echo $MODEL_NAME
echo $DIR

seed_value=42

python base.py \
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
  --num_train_epochs 10 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --auto_find_batch_size True \
  --logging_steps 10 \
  --load_best_model_at_end True \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  2> "experiments/$DIR/error.txt" > "experiments/$DIR/output.txt"

echo ".............. ending .............."
