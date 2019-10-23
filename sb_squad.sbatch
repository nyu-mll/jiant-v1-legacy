#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jiant-nq
#SBATCH --output=/scratch/pmh330/jiant-outputs/sb_jiant_nq.out
#SBATCH --error=/scratch/pmh330/jiant-outputs/sb_jiant_nq.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

#module purge # unload all modules
#module load cuda/8.0.44
#module load cudnn/8.0v5.1

source activate jiant
python main.py --config_file jiant/config/nq_bert.conf >> /scratch/pmh330/jiant-outputs/out_train_nq_v2.log 2>&1
