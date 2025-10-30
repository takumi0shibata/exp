#!/bin/bash

NUM_PAIRS=5000
# MODEL="DEV-000000-RandD-013-202405-gpt-5"
MODEL="DEV-000000-RandD-013-202405-gpt-5-mini"
# MODEL="DEV-000000-RandD-013-202405-gpt-5-nano"
MAX_WORKERS=5
MAX_RETRIES=6
BASE_SLEEP=1

for seed in 12 22 32 42 52
do
  for p in {1..8}
  do
    ATTS=$(python -c 'from src.utils.helper import target_attribute as t; import sys; print(" ".join(t(int(sys.argv[1]))))' "$p")
    for att in $ATTS;
    do
      echo "running: prompt=$p att=$att"
      python src/01_annotation.py \
        --prompt $p \
        --target-att $att \
        --num-pairs $NUM_PAIRS \
        --seed $seed \
        --model $MODEL \
        --max-workers $MAX_WORKERS \
        --max-retries $MAX_RETRIES \
        --base-sleep $BASE_SLEEP
    done
  done
done