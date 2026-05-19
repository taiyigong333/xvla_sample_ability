#!/bin/bash

# 基础路径（你可以根据自己环境修改）
BASE_ANALYSIS="F:\research\vla_team\X-VLA\analysis"
BASE_PYTHON="python"
PY_FILE="$BASE_ANALYSIS/occlusion_saliency_predict.py"
MAPPING_FILE="$BASE_ANALYSIS/config/mapping.json"
SERVER_URL="http://127.0.0.1:8010/act"

# 固定参数
GRID_SIZE=4
VIEWS="main wrist"
OCCLUSION="mean"
EVAL_STEPS=10
SEED=12345
CONTEXT_POLICY="around_close"
LIMIT_EPISODES=1

# ===================== 实验组合 =====================
# 数据集：black=v5, white=v6
# LoRA：10,20,30

declare -A DATASETS=(
    ["black"]="F:\data\data\_data\raw_demos_v5"
    ["white"]="F:\data\data\_data\raw_demos_v6"
)

LORA_VERSIONS=("10" "20" "30")

# ===================== 开始批量运行 =====================
for DATA_NAME in "${!DATASETS[@]}"; do
    DEMOS_DIR="${DATASETS[$DATA_NAME]}"
    
    for LORA_VER in "${LORA_VERSIONS[@]}"; do
        LORA_DIR="F:\research\vla_team\X-VLA\runnings\w_all_b_${LORA_VER}\ckpt-5000"
        OUT_DIR="$BASE_ANALYSIS/occlusion_saliency/${DATA_NAME}_lora${LORA_VER}"
        
        echo "=================================================="
        echo "🚀 运行：$DATA_NAME | LoRA=$LORA_VER"
        echo "📂 输出：$OUT_DIR"
        echo "=================================================="

        $BASE_PYTHON "$PY_FILE" \
        --demos "$DEMOS_DIR" \
        --out-dir "$OUT_DIR" \
        --mapping "$MAPPING_FILE" \
        --lora "$LORA_DIR" \
        --server-url "$SERVER_URL" \
        --limit-episodes $LIMIT_EPISODES \
        --grid-size $GRID_SIZE \
        --views $VIEWS \
        --occlusion $OCCLUSION \
        --eval-steps $EVAL_STEPS \
        --seed $SEED \
        --context-policy $CONTEXT_POLICY

        echo -e "\n✅ 完成：$DATA_NAME lora$LORA_VER\n\n"
    done
done

echo "🎉 所有对照实验全部完成！"