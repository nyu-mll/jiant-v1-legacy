#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=8000
#SBATCH --time=96:00:00
#SBATCH --job-name="nonjiant_cpu_job"
#SBATCH -o /home/zp489/runlogs/jobs/s%A.out

# Environment
export JIANT_DATA_DIR=/home/${USER}/data
source activate jiant

# Arguments
echo "==================="
echo "Args:"
echo JIANT_PROJECT_PREFIX="${JIANT_PROJECT_PREFIX}"
echo "==================="

# Run
cd /home/zp489/jiant
python $PROG.py $ARGS
