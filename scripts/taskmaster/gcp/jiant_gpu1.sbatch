#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=96:00:00
#SBATCH --job-name="jiant_job"

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
python \
    main.py \
    --config jiant/config/taskmaster/clean_roberta.conf \
    --overrides "${JIANT_OVERRIDES}"