#!/bin/bash

MODEL_DIR=$1 # directory of checkpoint to probe, e.g: /nfs/jsalt/share/models_to_probe/nli_do2_noelmo
EXP_NAME=$2
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"

# Note: you should only be overriding run_name and target_tasks and (maybe) something like probe_path. 

# (Alexis) Implicatives, Factives, Neutrals
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , target_tasks = nli-prob, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/to/pospos, nli-prob_use_classifier=mnli, nli-prob {probe_path = /home/alexis_ross_college_harvard_edu/working/to_pospos.csv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , target_tasks = nli-prob, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/to/negpos, nli-prob_use_classifier=mnli, nli-prob {probe_path = /home/alexis_ross_college_harvard_edu/working/to_negpos.csv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , target_tasks = nli-prob, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/that/pospos, nli-prob_use_classifier=mnli, nli-prob {probe_path = /home/alexis_ross_college_harvard_edu/working/that_pospos.csv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , target_tasks = nli-prob, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/that/negpos, nli-prob_use_classifier=mnli, nli-prob {probe_path = /home/alexis_ross_college_harvard_edu/working/that_negpos.csv}"

