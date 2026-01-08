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
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ${DATASET_DIR}/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${TAG}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens ${RETAINED_TOKENS} ${EXTRA_ARG}

python scripts/convert_answer_to_mme.py \
    --answer ./playground/data/eval/MME/answers/${TAG}.jsonl \
    --result_path ./playground/data/eval/MME/eval_tool/answers/${TAG} \
    --data_path ${DATASET_DIR}/MME/MME_Benchmark_release_version

cd ./playground/data/eval/MME/eval_tool

python calculation.py --results_dir answers/${TAG}
