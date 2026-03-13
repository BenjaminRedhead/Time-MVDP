#!/bin/bash
set -e

# =====================================================================
# Multivariate TimeDP Experiment
#
# Phase 1: Train univariate base TimeDP on 12 domains
# Phase 2: Fine-tune multivariate adapters on Exchange (8 currencies)
# Phase 3: Evaluate + generate report figures
#
# Run from: /workspace/TimeCraft/TimeDP/
# =====================================================================

export WANDB_MODE=disabled
export DATA_ROOT="/workspace/TimeCraft/TimeDP/data/training/TimeDP-Data"

DATASET="weather"
DATASET_CSV="/workspace/TimeCraft/TimeDP/data/${DATASET}/${DATASET}.csv"
SAVE_DIR="/workspace/writeable/TimeCraft/TimeDP"
SEQ_LEN=96
NUM_LATENTS=16
BASE_STEPS=50000
ADAPTER_STEPS=5000

mkdir -p ${SAVE_DIR}

# =====================================================================
# PHASE 1: Train univariate base model (uncomment when needed)
# =====================================================================
# echo "============================================================"
# echo "PHASE 1: Train univariate base model (all 12 domains)"
# echo "  Steps: ${BASE_STEPS}"
# echo "============================================================"
#
# PHASE1_START=$(date +%s)
#
# python main_train.py \
#     -b configs/multi_domain_timedp.yaml \
#     -up \
#     -sl ${SEQ_LEN} \
#     -nl ${NUM_LATENTS} \
#     -bs 128 \
#     -lr 0.001 \
#     --gpus 0, \
#     -l ${SAVE_DIR}/logs \
#     --no-test True \
#     --max_steps ${BASE_STEPS}
#
# PHASE1_END=$(date +%s)
# PHASE1_TIME=$((PHASE1_END - PHASE1_START))
# echo "Phase 1 completed in ${PHASE1_TIME}s ($((PHASE1_TIME / 60)) min)"

# Find checkpoint and log directory
BASE_CKPT=$(find ${SAVE_DIR}/logs -name "last.ckpt" -path "*/checkpoints/*" | sort | tail -1)
BASE_LOG_DIR=$(dirname $(dirname ${BASE_CKPT}))

echo "  Checkpoint: ${BASE_CKPT}"
echo "  Log dir: ${BASE_LOG_DIR}"

if [ -z "$BASE_CKPT" ]; then
    echo "ERROR: No checkpoint found."
    exit 1
fi

# =====================================================================
echo ""
echo "============================================================"
echo "PHASE 2: Fine-tune multivariate adapters on ${DATASET}"
echo "  Steps: ${ADAPTER_STEPS}"
echo "  Base checkpoint: ${BASE_CKPT}"
echo "============================================================"

PHASE2_START=$(date +%s)

python train_adapters.py \
    --base_ckpt "${BASE_CKPT}" \
    --dataset_csv "${DATASET_CSV}" \
    --config configs/multi_domain_timedp.yaml \
    --save_dir ${SAVE_DIR}/multivariate/${DATASET} \
    --seq_len ${SEQ_LEN} \
    --num_latents ${NUM_LATENTS} \
    --adapter_top_k 12 \
    --batch_size 4 \
    --lr 1e-4 \
    --n_steps ${ADAPTER_STEPS}

PHASE2_END=$(date +%s)
PHASE2_TIME=$((PHASE2_END - PHASE2_START))
echo ""
echo "Phase 2 completed in ${PHASE2_TIME}s ($((PHASE2_TIME / 60)) min)"

# =====================================================================
echo ""
echo "============================================================"
echo "PHASE 3: Evaluate + generate figures"
echo "============================================================"

ADAPTER_CKPT="${SAVE_DIR}/multivariate/${DATASET}/checkpoints/best.ckpt"
if [ ! -f "$ADAPTER_CKPT" ]; then
    ADAPTER_CKPT="${SAVE_DIR}/multivariate/${DATASET}/checkpoints/last.ckpt"
fi

PHASE3_START=$(date +%s)

python evaluate_multivariate.py \
    --ckpt "${ADAPTER_CKPT}" \
    --dataset_csv "${DATASET_CSV}" \
    --config configs/multi_domain_timedp.yaml \
    --save_dir ${SAVE_DIR}/multivariate/${DATASET} \
    --base_log_dir "${BASE_LOG_DIR}" \
    --seq_len ${SEQ_LEN} \
    --num_latents ${NUM_LATENTS} \
    --n_gen 100 \
    --ddim_steps 50 \
    --display_k 5

PHASE3_END=$(date +%s)
PHASE3_TIME=$((PHASE3_END - PHASE3_START))

# =====================================================================
echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "Timing:"
echo "  Phase 2 (adapters, 51K params):   ${PHASE2_TIME}s ($((PHASE2_TIME / 60)) min)"
echo "  Phase 3 (evaluation + figures):   ${PHASE3_TIME}s ($((PHASE3_TIME / 60)) min)"
echo ""
echo "Outputs: ${SAVE_DIR}/multivariate/${DATASET}/"
echo "  checkpoints/                — trained weights"
echo "  evaluation_results.json     — metrics"
echo "  adapter_timing.json         — Phase 2 timing"
echo "  figures/"
echo "    convergence_combined.pdf  — base + adapter training curves"
echo "    training_efficiency.pdf   — parameter/time comparison"
echo "    generated_vs_real.pdf"
echo "    correlation_matrices.pdf"
echo "    adjacency_graph.pdf"
echo "    results_comparison.pdf"