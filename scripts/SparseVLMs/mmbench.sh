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

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${DATASET_DIR}/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${TAG}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens ${RETAINED_TOKENS} ${EXTRA_ARG}

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATASET_DIR}/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment ${TAG}
