#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:1
#SBATCH --job-name=ccg
#SBATCH --output=slurm_%j.out


#load bertmnli+all npi, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_all_cola_npi, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_all_cola_npi_best.th\""

#load bertmnli+all npi with adv being held out, train and eval on adv probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_adv, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_adv_best.th\", target_tasks = \"npi_adv_li,npi_adv_sc,npi_adv_pr\""

#load bertmnli+all npi with cond being held out, train and eval on cond probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_cond, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_cond_best.th\", target_tasks = \"npi_cond_li,npi_cond_sc,npi_cond_pr\""