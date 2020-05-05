PROG="probing/retokenize_edge_data" ARGS="-t roberta-large /scratch/hl3236/data/edges/spr1/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large /scratch/hl3236/data/edges/spr2/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large /scratch/hl3236/data/edges/dpr/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large /scratch/hl3236/data/edges/dep_ewt/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large /scratch/hl3236/data/edges/ontonotes/const/pos/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large /scratch/hl3236/data/edges/ontonotes/const/nonterminal/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large /scratch/hl3236/data/edges/ontonotes/srl/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large /scratch/hl3236/data/edges/ontonotes/ner/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large /scratch/hl3236/data/edges/ontonotes/coref/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large /scratch/hl3236/data/edges/semeval/*.json" sbatch scripts/taskmaster/gcp/cpu.sbatch
PROG="scripts/ccg/align_tags_to_bert" ARGS="-t roberta-large -d /scratch/hl3236/data/ccg" sbatch scripts/taskmaster/gcp/cpu.sbatch
