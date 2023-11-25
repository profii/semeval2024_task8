DIR='allenai'

# DIR='roberta'
# DIR='xlm'

# DIR='numind'
# DIR='dslim'



python scorer.py \
  --gold_file_path "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/data/subtaskC_dev.jsonl" \
  --pred_file_path "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/predictions/subtaskC_dev.jsonl" \
  2> "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/result.txt" \
#   > "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/logs.txt"


echo " ending "

