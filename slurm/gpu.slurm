#!/bin/bash -l
# Run ANY step on one GPU. Everything after the script name is passed on:
#   sbatch slurm/gpu.slurm 05_train.py --model unet --run-name unet_base
#   sbatch slurm/gpu.slurm 13_predict_area.py --run unet_final --city leuven
# Submit from the code folder (the one with env.sh). Logs go to logs/<jobname>_<jobid>.out
#
# TODO before the first use:
#   --account   : the credit account of the course (the one you used in the labs, lp_...)
#   --partition : the GPU partition you used in the labs (wICE: gpu_a100 / gpu_h100; ask your TA)
#   --time      : a full U-Net run on ~5,000 images should take well under the limit below;
#                 check the duration of your first real run and adjust (shorter jobs start sooner).
#SBATCH --job-name=solar
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
source env.sh
echo "Running: python $*"
python "$@"
