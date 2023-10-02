source $HOME/supercloud_util/conda_env_tools.bash
#install the above

setup_worker_mamba open_flamingo

if [ ! $? -eq 0 ]; then
    echo 'ERROR: problem syncing env to worker'
    return 1
fi

mamba activate open_flamingo


export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u -m torch.distributed.run \
    --nproc_per_node 2 \
    --nnodes $COUNT_NODE \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --node_rank $SLURM_PROCID \
    /home/gridsan/omoll/repos/open_flamingo/open_flamingo/train/train.py \
    --lm_path "anas-awadalla/mpt-1b-redpajama-200b-hf-style" \
    --tokenizer_path "anas-awadalla/mpt-1b-redpajama-200b-hf-style" \
    --cross_attn_every_n_layers 4 \
    --dataset_resampled \
    --batch_size_mmc4 20 \
    --batch_size_laion 20 \
    --learning_rate 4e-5 \
    --train_num_samples_mmc4 120 \
    --train_num_samples_laion 120 \
    --loss_multiplier_laion 0.2 \
    --precision fp16 \
    --workers=10 \
    --deepspeed \
    --deepspeed_stage 2 \
    --wandb_project "open-flamingo-exp" \
    --report_to_wandb \
    --run_name "/home/gridsan/omoll/mpt-7b-deepspeed-job-${SLURM_JOB_ID}" \
    --num_epochs 10 \
    --warmup_steps 2 \
    --mmc4_textsim_threshold 0.24 \
    --laion_shards  "/home/gridsan/omoll/laion_dl/laion400m-data/00000.tar" \
    --mmc4_shards "/home/gridsan/omoll/mmc4_dl/mmc4_openflamingo4/000000000.tar" \
    --offline