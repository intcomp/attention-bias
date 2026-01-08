#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
WEIGHT=${WEIGHT:=""}

LAYER_LIST=${LAYER_LIST:=[2,10,20]}
RATIO_LIST=${RATIO_LIST:=[0.32,0.16,0.08]}

python -m PyramidDrop.llava.eval.model_vqa_mmbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${DATASET_DIR}/mmbench/mmbench_dev_20230712.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --layer_list  ${LAYER_LIST} \
    --image_token_ratio_list ${RATIO_LIST} \
    --conv-mode vicuna_v1 \
    --tag ${TAG} \
    --weight-file ${WEIGHT}

mkdir -p playground/data/eval/mmbench/answers_upload/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATASET_DIR}/mmbench/mmbench_dev_20230712.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST} \
    --experiment result_${TAG}
