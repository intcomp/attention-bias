#!/bin/bash

CKPT=llava-v1.5-7b
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
RETAINED_TOKENS="128"
SUFFIX="$1"
TAG="${CKPT}-${RETAINED_TOKENS}${SUFFIX}"

EXTRA_ARG=""
if [[ "$SUFFIX" == *no-padding* ]]
then
    EXTRA_ARG="--rm-padding"
fi
if [[ "$SUFFIX" == *weight-fit* ]]
then
    WEIGHT_PATH=${WEIGHT:="playground/data/weight/SparseVLMs/weights.npy"}
    EXTRA_ARG="${EXTRA_ARG} --weight-path ${WEIGHT_PATH}"
fi

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ${DATASET_DIR}/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${TAG}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens ${RETAINED_TOKENS} ${EXTRA_ARG}

python SparseVLMs/llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/${TAG}.jsonl
