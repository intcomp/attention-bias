#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
rank_list=${rank_list:=64}  # token_carve_image_token_rank = int(args.token_carve_image_token_nums / 2 * 3)
Ks=${Ks:=2}                 # token_carve_merge_token_nums = int(args.token_carve_image_token_nums / 2)
WEIGHT=${WEIGHT:=None}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m TokenCarve.TokenCarve.TokenCarve_model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
        --image-folder ${DATASET_DIR}/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/${CKPT}/TokenCarve/r_${rank_list}/k_${Ks}/${CHUNKS}_${IDX}_${TAG}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --use-token-carve True \
        --token-carve-sys-length 36 \
        --token-carve-image-token-length 576 \
        --token-carve-work-layer ${Ks} \
        --token-carve-image-token-nums ${rank_list} \
        --token-carve-SV-AV-mode 0 \
        --token-carve-SV-AV-weight 0.5 \
        --tag ${TAG} \
        --weight-file ${WEIGHT} &
done

wait

VQAV2_DIR="./playground/data/eval/vqav2"
output_file=./playground/data/eval/vqav2/answers/${CKPT}/TokenCarve/r_${rank_list}/k_${Ks}/merge_${TAG}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/${CKPT}/TokenCarve/r_${rank_list}/k_${Ks}/${CHUNKS}_${IDX}_${TAG}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py \
    --dir ${VQAV2_DIR} \
    --src answers/${CKPT}/TokenCarve/r_${rank_list}/k_${Ks}/merge_${TAG}.jsonl \
    --dst answers_upload/${CKPT}/TokenCarve/r_${rank_list}/k_${Ks}_${TAG}.json
