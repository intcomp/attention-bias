#!/bin/bash
 
CKPT=${CKPT:="llava-v1.5-7b"}
CKPT_DIR=${CKPT_DIR:="pretrained"}
DATASET_DIR=${DATASET_DIR:="dataset/playground/data/eval"}
rank_list=${rank_list:=128}
Ks=${Ks:=2}
WEIGHT=${WEIGHT:=""}

python -m FastV.src.FastV.inference.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${DATASET_DIR}/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use-fast-v True \
    --fast-v-sys-length 36 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank ${rank_list} \
    --fast-v-agg-layer ${Ks} \
    --tag ${TAG} \
    --weight-file ${WEIGHT}
    
python -m FastV.src.LLaVA.llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}/FastV/r_${rank_list}/k_${Ks}_${TAG}.jsonl
