#!/bin/bash

CKPT=llava-v1.5-7b
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
RETAINED_TOKENS=128
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
    EXTRA_ARG="${EXTRA_ARG} --weight-path ${WEIGHT_PATH}/weight/weights_exp.npy"
fi

# CUDA_VISIBLE_DEVICES=7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder ${DATASET_DIR}/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$TAG/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --retained_tokens ${RETAINED_TOKENS} ${EXTRA_ARG} &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$TAG/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$TAG/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --TAG $TAG
