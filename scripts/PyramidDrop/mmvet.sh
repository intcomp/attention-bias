#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
WEIGHT=${WEIGHT:=""}

LAYER_LIST=${LAYER_LIST:=[2,10,20]}
RATIO_LIST=${RATIO_LIST:=[0.32,0.16,0.08]}

python -m PyramidDrop.llava.eval.model_vqa \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ${DATASET_DIR}/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.jsonl \
    --temperature 0 \
    --layer_list  ${LAYER_LIST} \
    --image_token_ratio_list ${RATIO_LIST} \
    --conv-mode vicuna_v1 \
    --tag ${TAG} \
    --weight-file ${WEIGHT}

mkdir -p ./playground/data/eval/mm-vet/answers_upload/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.jsonl \
    --dst ./playground/data/eval/mm-vet/answers_upload/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.json

