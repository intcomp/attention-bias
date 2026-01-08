#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
WEIGHT=${WEIGHT:=None}

TXT_LAYER=${TXT_LAYER:=2}
TXT_RATIO=${TXT_RATIO:=115}
IMG_LAYER=${IMG_LAYER:=8}
IMG_RATIO=${IMG_RATIO:=90}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m HiMAP.src.HiMAP.inference.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
        --image-folder ${DATASET_DIR}/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$CKPT/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/${CHUNKS}_${IDX}_${TAG}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
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
        --weight-file ${WEIGHT} &
done

wait

GQA_DIR="./playground/data/eval/gqa/data"
output_file=./playground/data/eval/gqa/answers/$CKPT/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/merge_${TAG}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$CKPT/HiMAP/rank${TXT_RATIO}/Ks${TXT_LAYER}/${CHUNKS}_${IDX}_${TAG}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQA_DIR/testdev_balanced_predictions.json

cd $GQA_DIR
python eval/eval.py --tier testdev_balanced
