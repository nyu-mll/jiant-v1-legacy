from sklearn.metrics import matthews_corrcoef
import os

#go over all folders

result_path = "/scratch/sfw268/seminar/env1"

outfile = open(os.path.join(result_path, 'results.tsv'), 'w')

for folder in os.listdir(result_path):
    exps = [x for x in os.listdir(os.path.join(result_path, folder)) if x not in ['vocab', 'results.tsv', 'tasks', 'preproc', 'embs.pkl']]
    for exp in exps:
        results = [x for x in os.listdir(os.path.join(result_path, folder, exp)) if '.tsv' in x]
        for result in results:
            infile = open(os.path.join(result_path, folder, exp, result), 'r').readlines()
            preds = [x.split("\t")[1] for x in infile]
            labels = [x.split("\t")[4][0] for x in infile]
            outfile.write(folder+'\t'+exp+'\t'+result+'\t'+str(matthews_corrcoef(preds, labels))+'\n')

