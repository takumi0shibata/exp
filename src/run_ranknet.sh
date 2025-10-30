#!/bin/bash

NUM_PAIRS=5000
# MODEL="DEV-000000-RandD-013-202405-gpt-5"
MODEL="DEV-000000-RandD-013-202405-gpt-5-mini"
# MODEL="DEV-000000-RandD-013-202405-gpt-5-nano"

for seed in 12
do
  for p in {1..8}
  do
    ATTS=$(python -c 'from src.utils.helper import target_attribute as t; import sys; print(" ".join(t(int(sys.argv[1]))))' "$p")
    for att in $ATTS;
    do
      echo "running: prompt=$p att=$att"
      python src/02_01_train_ranknet.py \
        --prompt $p \
        --attribute $att \
        --num_pairs $NUM_PAIRS \
        --seed $seed \
        --model $MODEL \
        --ot_calibration
    done
  done
done