#!/bin/bash

#for n in 5 10 20

#do
#    python -u train_aug.py -freeze_emb -generation -aug_size $n -run_name asdiv_gts_aug_$n &
#    python -u train_aug.py -generation -aug_size $n -run_name asdiv_base_aug_$n &
#    python -u train_aug.py -epochs 40 -emb_name roberta-large -embedding_size 1024 -generation -aug_size $n -run_name asdiv_large_aug_$n

#done

#!/bin/bash

#for n in 5 10 20

#do
#    python -u train_aug_asdiv.py -freeze_emb -generation -aug_size $n -run_name asdiv_gts_aug_$n &
#    python -u train_aug_asdiv.py -generation -aug_size $n -run_name asdiv_base_aug_$n &
#    python -u train_aug_asdiv.py -epochs 40 -emb_name roberta-large -embedding_size 1024 -generation -aug_size $n -run_name asdiv_large_aug_$n 
#done

for n in 0 0.5 1

do
    python -u train_aug.py -val_size 2 -freeze_emb -generation -aug_size 0 -generation_threshold $n -run_name asdiv_gts_generation_$n &
    python -u train_aug.py -val_size 2 -generation -aug_size 0 -generation_threshold $n -run_name asdiv_base_generation_$n &
    python -u train_aug.py -val_size 2 -epochs 40 -emb_name roberta-large -embedding_size 1024 -generation -aug_size 0 -generation_threshold $n -run_name asdiv_large_generation_$n 
done