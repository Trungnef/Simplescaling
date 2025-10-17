#!/bin/bash
#SBATCH --job-name=finetune_qwen7b
#SBATCH --output=logs/finetune_qwen7b_%j.out
#SBATCH --nodelist=node001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# === Environment setup ===
source ~/.bashrc
conda activate env_llm

# === Training parameters ===
lr=1e-5
epochs=3
batch_size=4
weight_decay=1e-4
train_dataset_name="s1K_tokenized"
uid="$(date +%Y%m%d_%H%M%S)"

# === Optional arguments ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --train_dataset_name) train_dataset_name="$2"; shift 2 ;;
        --uid) uid="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# === GPU and node info ===
gpu_count=$(nvidia-smi -L | wc -l)
nnodes=1
head_node_ip=$(hostname -I | awk '{print $1}')
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export CUDA_VISIBLE_DEVICES=0

# === Gradient accumulation calculation ===
grad_acc=$((batch_size / (gpu_count * nnodes)))
if [ $grad_acc -lt 1 ]; then
  grad_acc=1
fi

echo "Running fine-tuning on $gpu_count GPU(s) with gradient_accumulation=$grad_acc"
echo "Head node IP: $head_node_ip"

# === Run training ===
run_name="qwen7b_${train_dataset_name}_bs${batch_size}_lr${lr}_epoch${epochs}_${uid}"

torchrun \
  --nnodes=$nnodes \
  --nproc_per_node=$gpu_count \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  train/sft.py \
  --block_size=16000 \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --gradient_accumulation_steps=$grad_acc \
  --num_train_epochs=${epochs} \
  --train_file_path="simplescaling/${train_dataset_name}" \
  --model_name="Qwen/Qwen2.5-7B-Instruct" \
  --warmup_ratio=0.05 \
  --report_to="none" \
  --bf16=True \
  --eval_strategy="no" \
  --logging_steps=10 \
  --save_strategy="steps" \
  --save_steps=1000 \
  --save_only_model=True \
  --gradient_checkpointing=True \
  --lr_scheduler_type="cosine" \
  --learning_rate=${lr} \
  --weight_decay=${weight_decay} \
  --output_dir="/data2/cmdir/home/ioit107/nmquy/LLMEducation/training/SFT/save_model/${run_name}"

echo "Training job submitted. Logs will be saved in logs/finetune_qwen7b_${SLURM_JOB_ID}.out"
