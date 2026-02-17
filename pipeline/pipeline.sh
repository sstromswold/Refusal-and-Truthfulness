#! /bin/sh

#SBATCH --job-name=refusal
#SBATCH --partition gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
module purge
module load cuda11.8/toolkit

PYTHONNOUSERSITE=1 conda run -p /home/common/neural-safety-net/refusal_direction/conda-ref python run_pipeline.py
