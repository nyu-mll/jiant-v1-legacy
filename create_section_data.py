

import pandas as pd
import re
from multiprocessing import Process, Manager
from tqdm import tqdm
from sklearn.model_selection import train_test_split
sections = ["FINAL DIAGNOSES", "CHIEF COMPLAINT", "FOLLOW-UP PLANS", "DISCHARGE STATUS", "DISCHARGE INSTRUCTIONS", "Followup Instructions", "DISCHARGE CONDITIO", "BRIEF SUMMARY OF HOSPITAL COURSE", "LABORATORY STUDIES", "PHYSICAL EXAM AT TIME OF ADMISSION", "SOCIAL HISTORY", "FAMILY HISTORY", "ALLERGIES", "MEDICATIONS ON ADMISSION", "PAST MEDICAL HISTORY", "HISTORY OF PRESENT ILLNESS"]

mappings = {"followup instructions": "discharge instructions"}
# Let's see all fo the preprocessin ghere. 
def assign_labels(csv_name):
	data = list(pd.read_csv(csv_name).T.to_dict().values())
	result = []
	def process_row(row, result):
		text = row["TEXT"]
		paragraphs = text.split("\n\n")
		paragraphs = paragraphs[::-1]
		current_text = []
		found_section = False
		for part in paragraphs[3:]:
			# we start at 3 becuase it ends iwth dictating who took the notes. s
			part = part.replace("\n", " ")
			found_section = False
			for section in sections:
				# we start at 3 becuase it ends iwth dictating who took the notes. 
				section = section.lower()
				if section in part.lower():
					# make sure it's the first one
					match = re.search(section, part.lower())
					index = match.start()
					try:
						if part[index + len(section)] == ":" or \
								part[index + len(section) + 1] == ":" or \
								part[index: index + len(section)].isupper():
							# Here, we found a section,not a word that is the section but is in the text.
							# only include the text following that index
							# there might be two that are in the sam
							non_section_text = part[index:]
							non_section_text = non_section_text.lower().replace(section+":", "")

							# get rid of any other secitons
							current_text.append(non_section_text)
							current_text.reverse()
							current_text = " ".join(current_text)
							result.append([current_text, section, row["ICD9_CODE"], row["TEXT"], row["HADM_ID"], row['CHARTDATE'], row['CHARTTIME'], row["CATEGORY"], row["ROW_ID"]])
							current_text = []
							found_section = True

					except Exception as e:
						# out-of-index error, meaning that the k section is at the end (which it never should if it is 
						# a section.
						print(e)
						continue
			if not found_section:
				# If tehre's a section thatisn't really reocngized than just ignore. 
				if ":" in part:
					# this was another oen we didn't catch	
					continue
				current_text.append(part)
							
		jobs = []
		# multiprocessing. 
	# this is with i = 3. 
	for i in range(len(data)):
		try:
			row = data[i]
			process_row(row, result)
			print("Finished processing %s" % i)
		except:
			continue
	result = pd.DataFrame(result, columns=["text", "section", "ICD9_CODE", "TEXT", "patient_id", "chart_date", "chart_time", "note_type", "ROW_ID"])
	return result

def make_train_test_split(dataframe):
	X = dataframe[["text", "section_text", "patient_id", "chart_date"]]
	y = dataframe[["section"]]
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
	test = pd.concat([X_test, y_test], axis=1)
	test.to_csv("section_test.tsv", sep="\t")
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y, test_size=0.2)
	train = pd.concat([X_train, y_train], axis=1)
	train.to_csv("section_train.tsv", sep="\t")
	val = pd.concat([X_val, y_val], axis=1)
	val.to_csv("section_val.tsv", sep="\t")
	print("The histogram of the section types")
	print(y_val["section"].value_counts(normalize=True))

def clean_chief_complaint():
	data = pd.read_csv("section_mapping.csv")
	data = list(data.T.to_dict().values())
	for row in data:
		if row["label"] == "chief complaint":
			try:
				row["text"] = ":".join(row["text"].split(":")[1:])
			except:
				continue
	data = pd.DataFrame(data, columns=["text", "section"])
	data.to_csv("section_mapping_final.csv")

result = assign_labels("icd_val.csv")
result.to_csv("icd_val_sections.csv")
import pdb; pdb.set_trace()
result = assign_labels("icd_test.csv")
result.to_csv("icd_test_sections.csv")
result = assign_labels("icd_train.csv")
result.to_csv("icd_train_sections.csv")



