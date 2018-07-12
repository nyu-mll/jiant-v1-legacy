#!/bin/bash

# DO NOT COMMIT CHANGES TO THIS FILE! Make a local copy and follow the
# instructions below.

# Copy this to /etc/profile.d/ to auto-set environment vars on login.
# Or, make a copy of this, customize, and run immediately before the training
# binary:
# cp path_config.sh ~/my_path_config.sh
# source ~/my_path_config.sh; python main.py --config ../config/demo.conf \
#   --overrides "do_train = 0"

# Default environment variables for JSALT code. May be overwritten by user.
# See https://github.com/jsalt18-sentence-repl/jiant for more.

##
# Example of custom paths for a local installation:
# export JIANT_PROJECT_PREFIX=/Users/Bowman/Drive/JSALT
# export JIANT_DATA_DIR=/Users/Bowman/Drive/JSALT/jiant/glue_data
# export WORD_EMBS_FILE=~/glove.840B.300d.txt
# export FASTTEXT_MODEL_FILE=None
# export FASTTEXT_EMBS_FILE=None

export JIANT_PROJECT_PREFIX=/nfs/jsalt/exp/nkim/vn
export JIANT_DATA_DIR=/nfs/jsalt/home/nkim
export NFS_PROJECT_PREFIX=/nfs/jsalt/exp/nkim/vn
export NFS_DATA_DIR=/nfs/jsalt/home/nkim
export MODEL_DIR=/nfs/jsalt/exp/nkim/models
export ELMO_MODEL=${MODEL_DIR}/mnli-elmo-do2-sd1/model_state_best.th
export NOELMO_MODEL=${MODEL_DIR}/mnli-noelmo-do2-sd1/model_state_main_epoch_23.best_macro.th
export ELMO_PARAMS_FILE=${MODEL_DIR}/mnli-elmo-do2-sd1/params.conf
export NOELMO_PARAMS_FILE=${MODEL_DIR}/mnli-noelmo-do2-sd1/params.conf

#echo "Loaded my config."



