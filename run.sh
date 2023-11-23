WANDB_PROJECT=nlp_hw2
echo $WANDB_PROJECT


MODEL_NAME='allenai/longformer-base-4096' # baseline
DIR='allenai'
# MODEL_NAME="bert-base-uncased" # bert-large-uncased !!!!!!add_prefix_space=False!!!!!
# MODEL_NAME='roberta-base' # roberta-large
# MODEL_NAME='xlm-roberta-base' # xlm-roberta-large-finetuned-conll03-english / xlm-roberta-large
# MODEL_NAME='bert-base-multilingual-cased'

# MODEL_NAME='numind/generic-entity_recognition_NER-v1'
# MODEL_NAME='dslim/bert-base-NER'
# MODEL_NAME='sentence-transformers/all-MiniLM-L6-v2'
# MODEL_NAME='jinaai/jina-embeddings-v2-base-en'


python my.py \
  --wandb $WANDB_PROJECT \
  --model_name $MODEL_NAME \
  --output_dir "experiments/$DIR" \
  --logging_dir "experiments/$MODEL_NAME/logs" \
  --num_epochs 20 \
  --save_total_limit 2 \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --logging_steps 10 \
  # --auto_find_batch_size False \
  # --output_dir "experiments/$MODEL_NAME" \
  # --save_steps 100 \
  # --add_prefix_space False \
#20
echo " ending "

