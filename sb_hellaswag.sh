#!/bin/bash
#SBATCH --job-name=xlm
#SBATCH --output=/misc/vlgscratch4/BowmanGroup/pmh330/jiant-outputs/tense-%j.out
#SBATCH --error=/misc/vlgscratch4/BowmanGroup/pmh330/jiant-outputs/tense-%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB
#SBATCH --signal=USR1@600
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1

module load anaconda3
source activate /misc/vlgscratch4/BowmanGroup/pmh330/conda/jiant
source user_config.sh
 
JIANT_CONF=$1
JIANT_OVERRIDES=$2
echo "$JIANT_CONF" 
echo "$JIANT_OVERRIDES"

/misc/vlgscratch4/BowmanGroup/pmh330/conda/jiant/bin/python main.py --config_file "jiant/config/taskmaster/base_roberta.conf" -o 'exp_name=xlm-roberta-large-mlm,pretrain_tasks="qqp,crosslingual_mlm", target_tasks="",reload_tasks=1, reload_indexing=0, reload_vocab=0, run_name=qqp_xlmr1, weighting_method=\"examples_proportional_mixingK=131072\", input_module=xlm-roberta-large,batch_size=4, do_target_task_training=0, transfer_paradigm=finetune, lr=2e-5, dropout=0.2, random_seed=111001, cuda="auto"'

#/misc/vlgscratch4/BowmanGroup/pmh330/conda/jiant/bin/python main.py --config_file "$JIANT_CONF" -o "$JIANT_OVERRIDES"
#python main.py --config_file jiant/config/mtl.conf --overrides "exp_name=multitask_mlm, run_name=mlm_sst$2,do_pretrain=1, do_target_task_training=0, lr=$1, dropout=0.2, random_seed=922"
