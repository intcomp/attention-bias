#!/bin/bash

CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
WEIGHT=${WEIGHT:=""}

LAYER_LIST=${LAYER_LIST:=[2,10,20]}
RATIO_LIST=${RATIO_LIST:=[0.32,0.16,0.08]}

python -m PyramidDrop.llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ${DATASET_DIR}/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.jsonl \
    --temperature 0 \
    --layer_list  ${LAYER_LIST} \
    --image_token_ratio_list ${RATIO_LIST} \
    --conv-mode vicuna_v1 \
    --tag ${TAG} \
    --weight-file ${WEIGHT}

python scripts/convert_answer_to_mme.py \
    --answer ./playground/data/eval/MME/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}.jsonl \
    --result_path ./playground/data/eval/MME/eval_tool/answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG} \
    --data_path ${DATASET_DIR}/MME/MME_Benchmark_release_version

cd ./playground/data/eval/MME/eval_tool

python calculation.py --results_dir answers/${CKPT}/PyramidDrop/rank${RATIO_LIST}/Ks${LAYER_LIST}/result_${TAG}
