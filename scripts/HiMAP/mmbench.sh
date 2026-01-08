#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
WEIGHT=${WEIGHT:=None}

TXT_LAYER=${TXT_LAYER:=2}
TXT_RATIO=${TXT_RATIO:=115}
IMG_LAYER=${IMG_LAYER:=8}
IMG_RATIO=${IMG_RATIO:=90}

python -m HiMAP.src.HiMAP.inference.model_vqa_mmbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${DATASET_DIR}/mmbench/mmbench_dev_20230712.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/result_${TAG}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --use-hmap-v \
    --sys-length 35 \
    --img-length 576 \
    --hmap-v-attn-txt-layer ${TXT_LAYER} \
    --hmap-v-attn-img-layer ${IMG_LAYER} \
    --hmap-v-attn-txt-rank ${TXT_RATIO} \
    --hmap-v-attn-img-rank ${IMG_RATIO} \
    --conv-mode vicuna_v1 \
    --tag ${TAG} \
    --weight-file ${WEIGHT}

mkdir -p playground/data/eval/mmbench/answers_upload/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATASET_DIR}/mmbench/mmbench_dev_20230712.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER} \
    --experiment result_${TAG}
