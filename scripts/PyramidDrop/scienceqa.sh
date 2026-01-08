#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
WEIGHT=${WEIGHT:=""}

LAYER_LIST=${LAYER_LIST:=[2,10,20]}
RATIO_LIST=${RATIO_LIST:=[0.32,0.16,0.08]}

python -m PyramidDrop.llava.eval.model_vqa_science \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ${DATASET_DIR}/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --layer_list  ${LAYER_LIST} \
    --image_token_ratio_list ${RATIO_LIST} \
    --conv-mode vicuna_v1 \
    --tag ${TAG} \
    --weight-file ${WEIGHT}

python PyramidDrop/llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_output_${TAG}.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_result_${TAG}.json
