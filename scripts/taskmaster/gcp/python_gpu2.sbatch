#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --mem=24000
#SBATCH --time=96:00:00
#SBATCH --job-name="nonjiant_job"
#SBATCH -o /home/zp489/runlogs/jobs/s%A.out

# Environment
export JIANT_DATA_DIR=/home/${USER}/data
export WORD_EMBS_FILE=
source activate jiant

# Arguments
echo "==================="
echo "Args:"
echo JIANT_PROJECT_PREFIX="${JIANT_PROJECT_PREFIX}"
echo JIANT_OVERRIDES="${JIANT_OVERRIDES}"
echo "==================="


# Run
cd /home/zp489/jiant
python $PROG.py $ARGS
