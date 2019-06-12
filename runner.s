#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=lion5

python main.py --config_file config/prpn.conf  --overrides "exp_name = my_exp_gpt, run_name = foobar, batch_size=5,  optimizer=sgd,  sent_enc=gpt_prpn, openai_transformer=1, word_embs=none, elmo=0, tokenizer=OpenAI.BPE, lr=0.1"
