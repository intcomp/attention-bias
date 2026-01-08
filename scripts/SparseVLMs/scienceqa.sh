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

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_science \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ${DATASET_DIR}/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${TAG}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens ${RETAINED_TOKENS} ${EXTRA_ARG}

python SparseVLMs/llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${TAG}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${TAG}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${TAG}_result.json
