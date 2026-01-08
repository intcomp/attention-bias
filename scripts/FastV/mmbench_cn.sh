#!/bin/bash
CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
rank_list=${rank_list:=128}
Ks=${Ks:=2}
WEIGHT=${WEIGHT:=""}

python -m FastV.src.FastV.inference.eval.model_vqa_mmbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${DATASET_DIR}/mmbench_cn/mmbench_dev_cn_20231003.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --lang cn \
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

mkdir -p playground/data/eval/mmbench_cn/answers_upload/${CKPT}/FastV/r_${rank_list}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_DIR}/mmbench_cn/mmbench_dev_cn_20231003.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/${CKPT}/FastV/r_${rank_list}\
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/${CKPT}/FastV/r_${rank_list} \
    --experiment k_${Ks}_${TAG}
