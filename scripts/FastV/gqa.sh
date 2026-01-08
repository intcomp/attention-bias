#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
rank_list=${rank_list:=128}
Ks=${Ks:=2}
WEIGHT=${WEIGHT:=""}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m FastV.src.FastV.inference.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
        --image-folder ${DATASET_DIR}/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}/${CHUNKS}_${IDX}_${TAG}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --use-fast-v True \
        --fast-v-sys-length 36 \
        --fast-v-image-token-length 576 \
        --fast-v-attention-rank ${rank_list} \
        --fast-v-agg-layer ${Ks} \
        --tag ${TAG} \
        --weight-file ${WEIGHT} &
done

wait

GQA_DIR="./playground/data/eval/gqa/data"
output_file=./playground/data/eval/gqa/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}/merge_${TAG}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}/${CHUNKS}_${IDX}_${TAG}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst ${GQA_DIR}/testdev_balanced_predictions.json

cd ${GQA_DIR}
python eval/eval.py \
    --tier testdev_balanced
