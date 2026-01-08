#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
rank_list=${rank_list:=128}
Ks=${Ks:=2}
WEIGHT=${WEIGHT:=""}

python -m FastV.src.FastV.inference.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ${DATASET_DIR}/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-fast-v True \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank ${rank_list} \
    --fast-v-agg-layer ${Ks} \
    --tag ${TAG} \
    --weight-file ${WEIGHT}

python scripts/convert_answer_to_mme.py \
    --answer ./playground/data/eval/MME/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --result_path ./playground/data/eval/MME/eval_tool/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG} \
    --data_path ${DATASET_DIR}/MME/MME_Benchmark_release_version

cd ./playground/data/eval/MME/eval_tool

python calculation.py --results_dir answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}
