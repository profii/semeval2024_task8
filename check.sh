# DIR='allenai' # 3, 6
# DIR='allenai_10ep_6hide' # 15.88
# DIR='allenai_15ep_18hide' # 12.89

# DIR='xlmr-lstm-crf' # 7.25
# DIR='xlmr-lstm-crf_2l_18hide' # 7.25
# DIR='xlmr-lstm-crf_2l_20ep' # 10.76

# DIR='xlm-roberta_2l_18hide' # 6.79
# DIR='xlm-roberta_2l_26hide' # 7.637
# DIR='xlm-roberta_2l_10head' # 7.637
# DIR='xlm-roberta_2l' # 10ep 7.63
# DIR='xlm-roberta_2l_gelun' # 7.637
# DIR='xlm-roberta_2l_key' # 10ep 7.637
# DIR='xlm-roberta_2l_keyquery' # 10ep 7.637
# DIR='crf_xlm-roberta_2l_18hide' # 7.887
# DIR='crf_xlm-roberta_20ep_2l_18hide' # 9.96
# DIR='xlm-roberta_2l_20ep_1drop' # 9.96

# DIR='xlm-roberta_2l_6hide' # 7.88
# DIR='xlm-roberta_f' # 10ep 7.88
# DIR='xlm-roberta_12ep_f' # 9.75
# DIR='xlm-roberta_5ep_f' # 9.96
# DIR='xlm-roberta_20ep_f' # 9.96
# DIR='dslim' # 20ep 8.07525
# DIR='dslim_f' # 20ep 8.07525
# DIR='xlm-fine_f' # 10 ep 10.78
# DIR='dslim' # 30ep 11.6
# DIR='dslim' # 10ep 11.7
# DIR='minilm' # 13
# DIR='minilm_20ep' # 14
# DIR='xlm_30ep' # 11.6
# DIR='xlm' # 15
# DIR='roberta_30ep' # 18.08
# DIR='roberta' # 18.5
# DIR='xlm-large_f' # 10 ep 19.64
# DIR='roberta_5ep' # 20


# DIR='xlm-roberta_drop' # 10 ep 31.28
# DIR='numind' #32
# DIR='roberta_f' # 38
# DIR='roberta_t' # 47



python scorer.py \
  --gold_file_path "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/data/subtaskC_dev.jsonl" \
  --pred_file_path "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/predictions/subtaskC_dev.jsonl" \
  2> "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/result.txt" \
#   > "/home/anastasiia.demidova/nlp_hw2/semeval2024_task8/experiments/$DIR/logs.txt"


echo " ending "

