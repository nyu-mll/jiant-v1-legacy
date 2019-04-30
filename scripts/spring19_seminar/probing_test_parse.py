from sklearn.metrics import matthews_corrcoef
import os

#go over all folders

result_path = "/scratch/sfw268/seminar/env1"

for file in os.listdir(result_path):
    exps = os.listdir(os.path.join(result_path, file))
    print(exps)

