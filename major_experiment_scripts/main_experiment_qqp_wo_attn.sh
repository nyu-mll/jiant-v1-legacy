# Use a run like this as a sanity check to make sure you can run with our current defaults on GPU.
# Change only the training_task name and exp_name.
python ../main.py --config ../config/test_run.conf --overrides "exp_name = test-run-qqp-alt, run_name = test, qqp-alt_pair_attn = 0, train_tasks = qqp-alt"

# Then, on Monday, run eight runs like these, changing only the training_task name and exp_name.
python ../main.py --config ../config/defaults.conf --overrides "exp_name = main-qqp-wo-attn-3, train_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = noelmo-do2-sd1, elmo_chars_only = 1, dropout = 0.2"
python ../main.py --config ../config/defaults.conf --overrides "exp_name = main-qqp-wo-attn-3, train_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = noelmo-do4-sd1, elmo_chars_only = 1, dropout = 0.4"
python ../main.py --config ../config/defaults.conf --overrides "exp_name = main-qqp-wo-attn-3, train_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = elmo-do2-sd1, elmo_chars_only = 0, dropout = 0.2"
python ../main.py --config ../config/defaults.conf --overrides "exp_name = main-qqp-wo-attn-3, train_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = elmo-do4-sd1, elmo_chars_only = 0, dropout = 0.4"
