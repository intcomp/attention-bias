#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
rank_list=${rank_list:=64}  # token_carve_image_token_rank = int(args.token_carve_image_token_nums / 2 * 3)
Ks=${Ks:=2}                 # token_carve_merge_token_nums = int(args.token_carve_image_token_nums / 2)
WEIGHT=${WEIGHT:=None}

python -m TokenCarve.TokenCarve.TokenCarve_model_vqa_science \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ${DATASET_DIR}/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${CKPT}/TokenCarve/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-token-carve True \
    --token-carve-sys-length 36 \
    --token-carve-image-token-length 576 \
    --token-carve-work-layer ${Ks} \
    --token-carve-image-token-nums ${rank_list} \
    --token-carve-SV-AV-mode 0 \
    --token-carve-SV-AV-weight 0.5 \
    --tag ${TAG} \
    --weight-file ${WEIGHT}

python -m TokenCarve.llava.eval.eval_science_qa \
    --base-dir ${DATASET_DIR}/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT}/TokenCarve/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}/TokenCarve/r_${rank_list}/k_${Ks}_output_${TAG}.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}/TokenCarve/r_${rank_list}/k_${Ks}_result_${TAG}.json
