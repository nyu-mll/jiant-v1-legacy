#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:1
#SBATCH --job-name=plains
#SBATCH --output=slurm_%j.out

python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_plain, load_target_train_checkpoint = \"/scratch/yc2552/exp/npi_bertccg/run_bertccg_model/model_state_pretrain_epoch_66.best_macro.th\", allow_untrained_encoder_parameters = 1"

python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_plain, load_target_train_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_pretrain_epoch_83.best_macro.th\", allow_untrained_encoder_parameters = 1"
