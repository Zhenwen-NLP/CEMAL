#!/bin/bash

#for n in 5 10 20

#do
#    python -u train_aug_svamp.py -freeze_emb -generation -aug_size $n -run_name svamp_gts_aug_$n &
#    python -u train_aug_svamp.py -generation -aug_size $n -run_name svamp_base_aug_$n &
#    python -u train_aug_svamp.py -epochs 40 -emb_name roberta-large -embedding_size 1024 -generation -aug_size $n -run_name svamp_large_aug_$n 
#done

for n in 0 0.5 1

do
    python -u train_aug_svamp.py -freeze_emb -generation -val_size 2 -aug_size 0 -generation_threshold $n -run_name svamp_gts_generation_$n &
    python -u train_aug_svamp.py -generation -val_size 2 -aug_size 0 -generation_threshold $n -run_name svamp_base_generation_$n &
    python -u train_aug_svamp.py -val_size 2 -epochs 40 -emb_name roberta-large -embedding_size 1024 -generation -aug_size 0 -generation_threshold $n -run_name svamp_large_generation_$n 
done