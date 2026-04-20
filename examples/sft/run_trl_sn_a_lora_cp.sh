#!/bin/bash
#PBS -N trl-sn
#PBS -q capacity
#PBS -l walltime=3:00:00
#PBS -l select=1
#PBS -l filesystems=flare
#PBS -A Intel-Aurora
#PBS -o trl_fsdp_sn_lora_8b_cp.log
#PBS -e trl_fsdp_sn_lora_8b_cp.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-${SCRIPT_DIR}}"

source ~/env-3.sh
module load xpu-smi/1.2.42
source /home/songhappy/miniforge3/etc/profile.d/conda.sh
conda activate trl2

export FI_PROVIDER_PATH=/home/xuehu/toolchains/2025.3.1.15/mpi/2021.17/opt/mpi/libfabric/lib/prov

# Single-node override.
export FI_PROVIDER=tcp
export FI_TCP_IFACE=lo
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=tcp

export TORCH_CPP_LOG_LEVEL=WARNING
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_DISTRIBUTED_CHECKPOINT_THREAD_COUNT=1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

RUN_NAME="trl_fsdp_sn_lora_8b_cp"
MODEL_PATH="/home/songhappy/models/Meta-Llama-3.1-8B-Instruct/"
OUTPUT_DIR="llama-sft-lora-fsdp-cp"
LOG_FILE="${PBS_O_WORKDIR:-${SCRIPT_DIR}}/${RUN_NAME}.log"
XPU_SMI_DUMP_FILE="${PBS_O_WORKDIR:-${SCRIPT_DIR}}/${RUN_NAME}_xpu_smi_dump.csv"
METRICS_FILE="${SCRIPT_DIR}/${RUN_NAME}_metrics.txt"
PLOTS_DIR="${SCRIPT_DIR}/plots"

XPU_SMI_PID=""
cleanup() {
    if [[ -n "${XPU_SMI_PID}" ]] && kill -0 "${XPU_SMI_PID}" >/dev/null 2>&1; then
        kill "${XPU_SMI_PID}" >/dev/null 2>&1 || true
        wait "${XPU_SMI_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

nohup xpu-smi dump -m 2,18 -i 1 > "${XPU_SMI_DUMP_FILE}" 2>&1 &
XPU_SMI_PID=$!

accelerate launch --config_file="${SCRIPT_DIR}/../accelerate_configs/fsdp2_lora_cp.yaml" \
    -m trl.scripts.sft_tokens \
    --model_name_or_path "${MODEL_PATH}" \
    --max_length 2048 \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 20 \
    --packing \
    --pad_to_multiple_of 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 10 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --report_to none \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir "${OUTPUT_DIR}"

cleanup
trap - EXIT

TRAINER_STATE_FILE="$(
    find "${OUTPUT_DIR}" -type f -name trainer_state.json \
        | sort -V \
        | tail -n 1
)"

if [[ -z "${TRAINER_STATE_FILE}" ]]; then
    echo "trainer_state.json not found under ${OUTPUT_DIR}" >&2
    exit 1
fi

python "${SCRIPT_DIR}/plot_trl_loss.py" \
    --trainer-state "${TRAINER_STATE_FILE}" \
    --log "${LOG_FILE}" \
    --out-dir "${PLOTS_DIR}" \
    --prefix "${RUN_NAME}"

python "${SCRIPT_DIR}/memo_perf_report.py" \
    --trainer-state "${TRAINER_STATE_FILE}" \
    --xpu-smi "${XPU_SMI_DUMP_FILE}" \
    --log "${LOG_FILE}" \
    | tee "${METRICS_FILE}"
