PROG="probing/retokenize_edge_data" ARGS="-t roberta-large ${JIANT_DATA_DIR}/edges/spr1/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large ${JIANT_DATA_DIR}/edges/spr2/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large ${JIANT_DATA_DIR}/edges/dpr/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large ${JIANT_DATA_DIR}/edges/dep_ewt/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large ${JIANT_DATA_DIR}/edges/ontonotes/const/pos/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large ${JIANT_DATA_DIR}/edges/ontonotes/const/nonterminal/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large ${JIANT_DATA_DIR}/edges/ontonotes/srl/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large ${JIANT_DATA_DIR}/edges/ontonotes/ner/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large ${JIANT_DATA_DIR}/edges/ontonotes/coref/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large ${JIANT_DATA_DIR}/edges/semeval/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="scripts/ccg/align_tags_to_bert" ARGS="-t roberta-large -d ${JIANT_DATA_DIR}/ccg" sbatch scripts/taskmaster/gcp/cpu.sbatch
