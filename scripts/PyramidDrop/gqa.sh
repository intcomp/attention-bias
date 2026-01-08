#!/bin/bash
CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
WEIGHT=${WEIGHT:=""}

LAYER_LIST=${LAYER_LIST:=[2,10,20]}
RATIO_LIST=${RATIO_LIST:=[0.32,0.16,0.08]}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m PyramidDrop.llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
        --image-folder ${DATASET_DIR}/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$CKPT/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/${CHUNKS}_${IDX}_${TAG}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --layer_list  ${LAYER_LIST} \
        --image_token_ratio_list ${RATIO_LIST} \
        --conv-mode vicuna_v1 \
        --tag ${TAG} \
        --weight-file ${WEIGHT} &
done

wait

GQADIR="./playground/data/eval/gqa/data"
output_file=./playground/data/eval/gqa/answers/$CKPT/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/merge_${TAG}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$CKPT/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/${CHUNKS}_${IDX}_${TAG}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
