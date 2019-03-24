python train.py \
  --voc_file_root ./data/voc/ \
  --data_dir ./data/ \
  --model_dir ./output/ \
  --enable_ema 0 \
  --res_number 10 \
  --enable_his_relevance_weight 0 \
  --history_strategy sum \
  --network_strategy mlp \
  --batch_size 32
