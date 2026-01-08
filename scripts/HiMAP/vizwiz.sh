#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
WEIGHT=${WEIGHT:=None}

TXT_LAYER=${TXT_LAYER:=2}
TXT_RATIO=${TXT_RATIO:=115}
IMG_LAYER=${IMG_LAYER:=8}
IMG_RATIO=${IMG_RATIO:=90}

python -m HiMAP.src.HiMAP.inference.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ${DATASET_DIR}/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/result_${TAG}.jsonl \
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

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/result_${TAG}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT}/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/result_${TAG}.json
