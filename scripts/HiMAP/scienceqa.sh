#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
WEIGHT=${WEIGHT:=None}

TXT_LAYER=${TXT_LAYER:=2}
TXT_RATIO=${TXT_RATIO:=115}
IMG_LAYER=${IMG_LAYER:=8}
IMG_RATIO=${IMG_RATIO:=90}

python -m HiMAP.src.HiMAP.inference.model_vqa_science \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ${DATASET_DIR}/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/result_${TAG}.jsonl \
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

python scripts/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/result_${TAG}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/result_output_${TAG}.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/result_result_${TAG}.json
