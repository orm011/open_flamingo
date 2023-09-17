#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=gpu:volta:2
#SBATCH --job-name train_deepspeed
#SBATCH -o %x_%j.log

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/home/gridsan/omoll/huggingface_cache/

export PYTHONFAULTHANDLER=1
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=15000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

srun --output=%x_%j_%t.log --nodes=${SLURM_NNODES} --ntasks=${SLURM_NTASKS} --cpu-bind=cores --accel-bind=gv \
    bash /home/gridsan/omoll/repos/open_flamingo/open_flamingo/scripts/run_node.bash