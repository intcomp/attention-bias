#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
WEIGHT=${WEIGHT:=""}

TASK=vizwiz
LAYER_LIST=${LAYER_LIST:=[2,10,20]}
RATIO_LIST=${RATIO_LIST:=[0.32,0.16,0.08]}

python -m PyramidDrop.llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ${DATASET_DIR}/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.jsonl \
    --temperature 0 \
    --layer_list  ${LAYER_LIST} \
    --image_token_ratio_list ${RATIO_LIST} \
    --conv-mode vicuna_v1 \
    --tag ${TAG} \
    --weight-file ${WEIGHT} 

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.json
