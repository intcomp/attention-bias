#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
rank_list=${rank_list:=128}
Ks=${Ks:=2}
WEIGHT=${WEIGHT:=""}

python -m FastV.src.FastV.inference.eval.model_vqa_loader_fastV \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ${DATASET_DIR}/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-fast-v True \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank ${rank_list} \
    --fast-v-agg-layer ${Ks} \
    --tag ${TAG} \
    --weight-file ${WEIGHT}

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}.json
