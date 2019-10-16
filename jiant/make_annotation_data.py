import os
import pandas as pd
from os import listdir

hey = pd.DataFrame(columns=["id", "text", "has_label"])
from os.path import isfile, join
current_dir = "/Users/yadapruksachatkun/Downloads/concept_assertion_relation_training_data/"
onlyfiles_beth = [os.path.join("beth/txt", f) for f in listdir(os.path.join(current_dir, "beth/txt")) if isfile(os.path.join(current_dir, "beth/txt", f))]
onlyfiles_partners = [os.path.join("partners/txt", f) for f in listdir(os.path.join(current_dir, "partners/txt")) if isfile(os.path.join(current_dir, "partners/txt", f))]
onlyfiles_beth_partners_unknown = [os.path.join("partners/unannotated", f) for f in listdir(os.path.join(current_dir, "partners/unannotated")) if isfile(os.path.join(current_dir, "partners/unannotated",f))]
i = 0
for dataset in [onlyfiles_beth, onlyfiles_partners, onlyfiles_beth_partners_unknown]:
	has_label = True
	i += 1
	if i == 3:
		has_label = False
	for file in dataset:
		try:
			lines = "\n".join(pd.read_csv(os.path.join(current_dir, file), header=None, sep="\t")[0].values)
			current = pd.DataFrame([[file, lines, has_label]], columns=["id", "text", "has_label"])
			hey = hey.append(current)
		except:
			import pdb; pdb.set_trace()
			continue
 
hey.to_csv("to_annotate_i2b2_va.csv")
