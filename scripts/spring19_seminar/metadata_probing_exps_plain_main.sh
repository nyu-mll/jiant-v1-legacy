#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:1
#SBATCH --job-name=plains
#SBATCH --output=slurm_%j.out

python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=bertccg_plain, run_name = main, load_target_train_checkpoint = \"/scratch/yc2552/exp/npi_bertccg/run_bertccg_model/model_state_pretrain_epoch_66.best_macro.th\", allow_untrained_encoder_parameters = 1, target_tasks = \"cola, cola_npi_sup, cola_npi_qnt, cola_npi_quessmp, cola_npi_ques, cola_npi_only, cola_npi_negsent, cola_npi_negdet, cola_npi_cond, cola_npi_adv\""

python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=bertmnli_plain, run_name = main, load_target_train_checkpoint = \"/scratch/yc2552/exp/npi_bertmnli/run_bertmnli_model/model_state_pretrain_epoch_83.best_macro.th\", allow_untrained_encoder_parameters = 1, target_tasks = \"cola, cola_npi_sup, cola_npi_qnt, cola_npi_quessmp, cola_npi_ques, cola_npi_only, cola_npi_negsent, cola_npi_negdet, cola_npi_cond, cola_npi_adv\""

python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=bert_plain, run_name = main, load_eval_checkpoint = none, allow_untrained_encoder_parameters = 1, target_tasks = \"cola, cola_npi_sup, cola_npi_qnt, cola_npi_quessmp, cola_npi_ques, cola_npi_only, cola_npi_negsent, cola_npi_negdet, cola_npi_cond, cola_npi_adv\""

python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=bow_glove_plain, run_name = main, load_eval_checkpoint = none, allow_untrained_encoder_parameters = 1, sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\", target_tasks = \"cola, cola_npi_sup, cola_npi_qnt, cola_npi_quessmp, cola_npi_ques, cola_npi_only, cola_npi_negsent, cola_npi_negdet, cola_npi_cond, cola_npi_adv\""

