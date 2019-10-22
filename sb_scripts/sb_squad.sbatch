#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jiant-squad
#SBATCH --output=/scratch/pmh330/jiant-outputs/sb_jiant_squad.out
#SBATCH --error=/scratch/pmh330/jiant-outputs/sb_jiant_squad.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

#module purge # unload all modules
#module load cuda/8.0.44
#module load cudnn/8.0v5.1

source activate jiant
python main.py --config_file jiant/config/squad_bert.conf >> /scratch/pmh330/jiant-outputs/out_train_squad_v4.log 2>&1
