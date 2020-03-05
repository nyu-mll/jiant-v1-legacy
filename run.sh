export JIANT_PROJECT_PREFIX="exps"
export JIANT_DATA_DIR="data"
export NFS_PROJECT_PREFIX=""	
export NFS_DATA_DIR=""
export WORD_EMBS_FILE="glove.840B.300d.txt"
export FASTTEXT_MODEL_FILE=None	
export FASTTEXT_EMBS_FILE=None	

python main.py --config jiant/config/demo.conf --overrides "exp_name=followups_final, max_vals=1, split=1, allow_missing_task_map = 1, write_preds=test, scaling_method=max_inverse, sent_enc=rnn,run_name=followups_binary, write_preds=\"val,test\",lr=0.001, max_seq_len=100000,val_interval=50, batch_size=4, tokenizer=MosesTokenizer,load_model=1, pretrain_tasks = \"followups_binary\", target_tasks=\"followups_binary\",input_module=glove, do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1"