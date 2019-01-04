# Scripts to recreate GLUE baselines

JIANT_OVERRIDES="do_target_task_training = 0, pretrain_tasks = qnliv2, target_tasks = qnliv2, qnliv2_pair_attn = 0, run_name = test, elmo_chars_only = 0" JIANT_CONF="config/glue.conf" sbatch nyu_cilvr_cluster.sbatch

## Single-task ##
# GloVe
JIANT_OVERRIDES="exp_name = glue-baselines-glove, run_name = qnli-single-glove-noattn, pretrain_tasks = qnliv2, target_tasks = qnliv2, do_target_task_training = 0, pair_attn = 0, word_embs = glove, elmo = 0, cuda = 5" JIANT_CONF="config/glue.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="exp_name = glue-baselines-glove, run_name = qnli-single-glove-attn, pretrain_tasks = qnliv2, target_tasks = qnliv2, do_target_task_training = 0, pair_attn = 1, word_embs = glove, elmo = 0, cuda = 7" JIANT_CONF="config/glue.conf" sbatch nyu_cilvr_cluster.sbatch

# CoVe
JIANT_OVERRIDES="exp_name = glue-baselines-glove, run_name = qnli-single-cove-noattn, pretrain_tasks = qnliv2, target_tasks = qnliv2, do_target_task_training = 0, pair_attn = 0, word_embs = glove, cove = 1, elmo = 0, cuda = 3" JIANT_CONF="config/glue.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="exp_name = glue-baselines-glove, run_name = qnli-single-cove-attn, pretrain_tasks = qnliv2, target_tasks = qnliv2, do_target_task_training = 0, pair_attn = 1, word_embs = glove, cove = 1, elmo = 0, cuda = 3" JIANT_CONF="config/glue.conf" sbatch nyu_cilvr_cluster.sbatch

# ELMo
JIANT_OVERRIDES="run_name = qnli-single-elmo-noattn, pretrain_tasks = qnliv2, target_tasks = qnliv2, do_target_task_training = 0, pair_attn = 0, elmo_chars_only = 0, cuda = 7" JIANT_CONF="config/glue.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="run_name = qnli-single-elmo-attn, pretrain_tasks = qnliv2, target_tasks = qnliv2, do_target_task_training = 0, pair_attn = 1, elmo_chars_only = 0, cuda = 5" JIANT_CONF="config/glue.conf" sbatch nyu_cilvr_cluster.sbatch

## Multi-task ##
JIANT_OVERRIDES="" JIANT_CONF="config/glue.conf" sbatch nyu_cilvr_cluster.sbatch
