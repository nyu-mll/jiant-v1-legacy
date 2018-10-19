#!/bin/bash

MODEL_DIR=$1 # directory of checkpoint to probe, e.g: /nfs/jsalt/share/models_to_probe/nli_do2_noelmo
PROBING_TASK=$2 # task name, e.g. recast-puns
RUN_NAME=${3:-"test"}

EXP_NAME="probing"
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"

# model=random runs=100 classifier=log reg
python main.py --config config/defaults.conf --overrides "exp_name = 100_MNLI/log_reg/, load_eval_checkpoint = /nfs/jsalt/share/models_to_probe/final/random-noelmo/model_state_eval_best.th, train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, eval_tasks = mnli, do_eval = 1, eval_data_fraction = 1, run_name = random-noelmo, elmo_chars_only = 1"

# model=grounded runs=100 classifier=log reg

python main.py --config config/defaults.conf --overrides "exp_name = 100_MNLI/log_reg/, load_eval_checkpoint = /nfs/jsalt/share/models_to_probe/final/grounded-noelmo/model_state_eval_best.th, train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, eval_tasks = mnli, do_eval = 1, eval_data_fraction = 1, run_name = grounded-noelmo, elmo_chars_only = 1"

# model=ccg runs=100 classifier=log reg

python main.py --config config/defaults.conf --overrides "exp_name = 100_MNLI/log_reg/, load_eval_checkpoint = /nfs/jsalt/share/models_to_probe/final/ccg-noelmo/model_state_eval_best.th, train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, eval_tasks = mnli, do_eval = 1, eval_data_fraction = 1, run_name = ccg-noelmo, elmo_chars_only = 1"

# model=nli runs=100 classifier=log reg

python main.py --config config/defaults.conf --overrides "exp_name = 100_MNLI/log_reg/, load_eval_checkpoint = /nfs/jsalt/share/models_to_probe/final/mnli-noelmo/model_state_eval_best.th, train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, eval_tasks = mnli, do_eval = 1, eval_data_fraction = 1, run_name = mnli-noelmo, elmo_chars_only = 1"

# model=wiki103 runs=100 classifier=log reg

python main.py --config config/defaults.conf --overrides "exp_name = 100_MNLI/log_reg/, load_eval_checkpoint = /nfs/jsalt/share/models_to_probe/final/wiki103-lm-noelmo/model_state_eval_best.th, train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, eval_tasks = mnli, do_eval = 1, eval_data_fraction =1, run_name = wiki103-lm-noelmo, elmo_chars_only = 1"

