#!/bin/bash

NUM_PAIRS=5000

# GPT
# MODEL="DEV-000000-RandD-013-202405-gpt-5"
# MODEL="DEV-000000-RandD-013-202405-gpt-5-mini"
# MODEL="DEV-000000-RandD-013-202405-gpt-5-nano"
MAX_WORKERS=5
MAX_RETRIES=6
BASE_SLEEP=1

# Gemma
MODEL="google/gemma-3n-e2b-it"
GEMMA_BATCH_SIZE=10
GEMMA_MAX_NEW_TOKENS=512

for seed in 12
do
  for p in {1..7}
  do
    ATTS=$(python -c 'from src.utils.helper import target_attribute as t; import sys; print(" ".join(t(int(sys.argv[1]))))' "$p")
    for att in $ATTS;
    do
      echo "running: prompt=$p att=$att"
      python src/01_annotation.py \
        --target-prompt $p \
        --target-att $att \
        --num-pairs $NUM_PAIRS \
        --seed $seed \
        --model $MODEL \
        --max-workers $MAX_WORKERS \
        --max-retries $MAX_RETRIES \
        --base-sleep $BASE_SLEEP \
        --gemma-batch-size $GEMMA_BATCH_SIZE \
        --gemma-max-new-tokens $GEMMA_MAX_NEW_TOKENS
    done
  done
done