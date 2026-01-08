export CKPT_DIR="pretrained" 
export DATASET_DIR="dataset/playground/data/eval"

TASKS=(
    "gqa"
    "mmbench_cn"
    "mmbench"
    "MME"
    "mmvet"
    "pope"
    "scienceqa"
    "textvqa"
    "vizwiz"
    "vqav2"
)

for TASK in "${TASKS[@]}"; do
    export TAG=after
    export WEIGHT="playground/data/weight/PyramidDrop/${TASK}.npy"
    echo "Running: Task=${TASK}"
    if scripts/PyramidDrop/${TASK}.sh; then
        echo "[Success]"
    else
        echo "[Error]"
    fi
done