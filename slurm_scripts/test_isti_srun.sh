#!/bin/bash -x
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=open_clip
#SBATCH --mem=60G
#SBATCH --output=slurm_out/distributed_test-%j.out

eval "$(/app/anaconda3/bin/conda shell.bash hook)" # init conda
conda activate open_clip_test
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd Repos/open_clip_test/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
	--save-frequency 1 \
    --wandb-project-name "self_sustain_clip" \
	--dataset-type "csv" \
	--dataset-resampled \
	--train-data "../root/mscoco/mscoco2014_train_rel_path.csv" \
	--warmup 2000 \
	--batch-size 32 \
	--epochs 21 \
	--workers 2 \
	--model "selfsustain_ViT-B-32" \
	--logs "isti_test" \
	--seed 0 \
	--local-loss \
	--gather-with-grad \
	--report-to "wandb" \
	--log-every-n-step 1 \
	--grad-checkpointing \
    --self-sustain \
    --self-sustain-oracle-name "sentence-transformers/all-MiniLM-L6-v2" \
    --self-sustain-lambda-start 7 \
    --self-sustain-lambda-end 14 \
    --self-sustain-oracle-max-len 77
    
