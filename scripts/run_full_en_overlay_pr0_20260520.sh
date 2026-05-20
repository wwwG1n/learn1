#!/bin/bash
set -e
source /home/kakakaaka/anaconda3/etc/profile.d/conda.sh
export PATH=/home/kakakaaka/anaconda3/bin:$PATH
cd /mnt/data1/yanfeihong/projs/Lumina-DiMOO_AdaTokenPruning
export CUDA_VISIBLE_DEVICES=0
export GENEVAL_OUTPUT_DIR_PREFIX=full_en_overlay_pr0
export GENEVAL_GEN_EXTRA_ARGS="--height 512 --width 512 --n_samples 4 --en_heatmap"
./scripts/eval_gen_eval.sh
