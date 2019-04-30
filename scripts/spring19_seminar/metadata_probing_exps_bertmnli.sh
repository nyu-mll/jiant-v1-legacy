#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:1
#SBATCH --job-name=mnli
#SBATCH --output=slurm_%j.out


#load bertmnli plain, train and eval on all probing tasks (MAY HAVE TO SUPPLY MODEL NAME MANUALLY)
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_plain, load_target_train_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_pretrain_epoch_83.best_macro.th\", allow_untrained_encoder_parameters = 1"

#load bertmnli+cola, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_cola, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_cola_best.th\""

#load bertmnli+all npi with negdet being held out, train and eval on negdet probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_negdet, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_negdet_best.th\", target_tasks = \"npi_negdet_li,npi_negsent_sc,npi_negsent_pr\""

#load bertmnli+all npi with negsent being held out, train and eval on negsent probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_negsent, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_negsent_best.th\", target_tasks = \"npi_negsent_li,npi_negsent_sc,npi_negsent_pr\""

#load bertmnli+all npi with only being held out, train and eval on only probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_only, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_only_best.th\", target_tasks = \"npi_only_li,npi_only_sc,npi_only_pr\""

#load bertmnli+all npi with qnt being held out, train and eval on qnt probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_qnt, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_qnt_best.th\", target_tasks = \"npi_qnt_li,npi_qnt_sc,npi_qnt_pr\""

#load bertmnli+all npi with ques being held out, train and eval on ques probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_ques, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_ques_best.th\", target_tasks = \"npi_ques_li,npi_ques_sc,npi_ques_pr\""

#load bertmnli+all npi with quessmp being held out, train and eval on quessmp probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_quessmp, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_quessmp_best.th\", target_tasks = \"npi_quessmp_li,npi_quessmp_sc,npi_quessmp_pr\""

#load bertmnli+all npi with sup being held out, train and eval on sup probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_sup, load_eval_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_sup_best.th\", target_tasks = \"npi_sup_li,npi_sup_sc,npi_sup_pr\""