#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
rank_list=${rank_list:=128}
Ks=${Ks:=2}
WEIGHT=${WEIGHT:=""}

python -m FastV.src.FastV.inference.eval.model_vqa_science \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ${DATASET_DIR}/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-fast-v True \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank ${rank_list} \
    --fast-v-agg-layer ${Ks} \
    --tag ${TAG} \
    --weight-file ${WEIGHT}

python -m FastV.src.LLaVA.llava.eval.eval_science_qa \
    --base-dir ${DATASET_DIR}/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_output_${TAG}.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_result_${TAG}.json
