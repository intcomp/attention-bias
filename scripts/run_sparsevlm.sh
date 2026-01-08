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
    echo "Running: Task=${TASK}"
    export WEIGHT="playground/data/weight/SparseVLMs/${TASK}.npy"
    if scripts/SparseVLMs/${TASK}.sh -no-padding-weight-fit; then
        echo "[Success]"
    else
        echo "[Error]"
    fi
done