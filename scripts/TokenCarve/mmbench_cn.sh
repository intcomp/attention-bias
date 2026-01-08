#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
rank_list=${rank_list:=64}  # token_carve_image_token_rank = int(args.token_carve_image_token_nums / 2 * 3)
Ks=${Ks:=2}                 # token_carve_merge_token_nums = int(args.token_carve_image_token_nums / 2)
WEIGHT=${WEIGHT:=None}

python -m TokenCarve.TokenCarve.TokenCarve_model_vqa_mmbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${DATASET_DIR}/mmbench_cn/mmbench_dev_cn_20231003.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/${CKPT}/TokenCarve/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --lang cn \
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

mkdir -p playground/data/eval/mmbench_cn/answers_upload/${CKPT}/TokenCarve/r_${rank_list}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATASET_DIR}/mmbench_cn/mmbench_dev_cn_20231003.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/${CKPT}/TokenCarve/r_${rank_list}\
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/${CKPT}/TokenCarve/r_${rank_list} \
    --experiment k_${Ks}_${TAG}
