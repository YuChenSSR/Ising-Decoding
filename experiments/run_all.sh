#!/usr/bin/env bash
# Batch experiment runner for CAL paper
# Usage: CUDA_VISIBLE_DEVICES=7 bash experiments/run_all.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="code:${PYTHONPATH:-}"

RESULTS_DIR="experiments/results"
mkdir -p "${RESULTS_DIR}"

FAST_MODEL="models/Ising-Decoder-SurfaceCode-1-Fast.pt"       # Model 1, R=9
ACCURATE_MODEL="models/Ising-Decoder-SurfaceCode-1-Accurate.pt" # Model 4, R=13

# Common inference args
BASE_ARGS="--config-name=config_public workflow.task=inference +exp_tag=bench ++load_checkpoint=True"

run_inference() {
    local label="$1"
    local extra_args="$2"
    local logfile="${RESULTS_DIR}/${label}.log"
    echo ">>> Running: ${label}"
    PREDECODER_ENABLE_TIMING_INSTRUMENTATION=1 \
    python -u code/workflows/run.py \
        ${BASE_ARGS} ${extra_args} 2>&1 | tee "${logfile}"
    echo ">>> Done: ${label} -> ${logfile}"
    echo ""
}

###############################################################################
# Experiment 1: Backend comparison (d=7, Model 1)
# PyTorch / PyTorch+compile / TensorRT FP16 / TensorRT INT8
###############################################################################
echo "=============================================="
echo "Experiment 1: Backend comparison (d=7, Model1)"
echo "=============================================="

# 1a. PyTorch (no compile)
PREDECODER_TORCH_COMPILE=0 ONNX_WORKFLOW=0 \
run_inference "exp1_pytorch_nocompile_d7" \
    "+model_checkpoint_file=${FAST_MODEL} distance=7 n_rounds=7"

# 1b. PyTorch + torch.compile
PREDECODER_TORCH_COMPILE=1 ONNX_WORKFLOW=0 \
run_inference "exp1_pytorch_compile_d7" \
    "+model_checkpoint_file=${FAST_MODEL} distance=7 n_rounds=7"

# 1c. TensorRT FP16
PREDECODER_TORCH_COMPILE=0 ONNX_WORKFLOW=2 \
run_inference "exp1_trt_fp16_d7" \
    "+model_checkpoint_file=${FAST_MODEL} distance=7 n_rounds=7"

# 1d. TensorRT INT8
PREDECODER_TORCH_COMPILE=0 ONNX_WORKFLOW=2 QUANT_FORMAT=int8 \
run_inference "exp1_trt_int8_d7" \
    "+model_checkpoint_file=${FAST_MODEL} distance=7 n_rounds=7"

###############################################################################
# Experiment 2: Quantization tradeoff (Model 1 R=9 + Model 4 R=13)
###############################################################################
echo "=============================================="
echo "Experiment 2: Quantization tradeoff"
echo "=============================================="

for model_label in fast accurate; do
    if [ "${model_label}" = "fast" ]; then
        MODEL_FILE="${FAST_MODEL}"
        MID="model1"
    else
        MODEL_FILE="${ACCURATE_MODEL}"
        MID="model4"
    fi

    # FP32 (PyTorch, no compile for fair comparison)
    PREDECODER_TORCH_COMPILE=0 ONNX_WORKFLOW=0 \
    run_inference "exp2_${MID}_fp32_d7" \
        "+model_checkpoint_file=${MODEL_FILE} distance=7 n_rounds=7"

    # TRT FP16
    PREDECODER_TORCH_COMPILE=0 ONNX_WORKFLOW=2 \
    run_inference "exp2_${MID}_trt_fp16_d7" \
        "+model_checkpoint_file=${MODEL_FILE} distance=7 n_rounds=7"

    # TRT INT8
    PREDECODER_TORCH_COMPILE=0 ONNX_WORKFLOW=2 QUANT_FORMAT=int8 \
    run_inference "exp2_${MID}_trt_int8_d7" \
        "+model_checkpoint_file=${MODEL_FILE} distance=7 n_rounds=7"
done

###############################################################################
# Experiment 3: Distance scaling (Model 1, TRT FP16)
###############################################################################
echo "=============================================="
echo "Experiment 3: Distance scaling"
echo "=============================================="

for d in 5 7 9 11 13 15 17 21; do
    PREDECODER_TORCH_COMPILE=0 ONNX_WORKFLOW=2 \
    run_inference "exp3_trt_fp16_d${d}" \
        "+model_checkpoint_file=${FAST_MODEL} distance=${d} n_rounds=${d}"
done

###############################################################################
# Experiment 4: Realtime analysis - same as Exp3 but also with baseline timing
# (already collected via PREDECODER_ENABLE_TIMING_INSTRUMENTATION=1)
###############################################################################
echo "=============================================="
echo "Experiment 4: INT8 distance scaling"
echo "=============================================="

for d in 5 7 9 11 13; do
    PREDECODER_TORCH_COMPILE=0 ONNX_WORKFLOW=2 QUANT_FORMAT=int8 \
    run_inference "exp4_trt_int8_d${d}" \
        "+model_checkpoint_file=${FAST_MODEL} distance=${d} n_rounds=${d}"
done

echo "=============================================="
echo "All experiments complete!"
echo "Results in: ${RESULTS_DIR}/"
echo "=============================================="
