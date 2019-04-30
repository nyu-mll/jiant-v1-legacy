import re

results = open("probing_results.tsv", "r").read()

results = re.sub("npiALL", "all_cola_npi", results)
results = re.sub("\tnpi_", "\t", results)
results = re.sub("hd[a-zA-Z_]+\t", "HoldOutNPI\t", results)
results = re.sub("NPI_probing_|_test.tsv|_glove", "", results)
results = re.sub("^bow_plain", "bow", results)
results = re.sub("\nbow_plain", "\nbow", results)
results = re.sub("\tbow_|\tbert_|\tbertccg_|\tbertmnli_", "\t", results)
results = re.sub("all_cola_npi", "AllNPI", results)
results = re.sub("_", "\t", results)


print(results)

results = results.split('\n')

results_dict = {}

for x in results:
    x = x.split('\t')
    print(x)
    if len(x) > 2:
        results_dict[x[0]] = results_dict.get(x[0], {}) 
        results_dict[x[0]][x[1]] = results_dict[x[0]].get(x[1], {}) 
        results_dict[x[0]][x[1]][x[3]] = results_dict[x[0]][x[1]].get(x[3], {}) 
        results_dict[x[0]][x[1]][x[3]][x[2]] = results_dict[x[0]][x[1]][x[3]].get(x[2], x[4]) 

outfile = open("parsed_results.txt", "w")
outfile.write("model")
#probing_type
for p in sorted(results_dict['bow']['plain'].keys()):
    #data set
    for d in sorted(results_dict['bow']['plain'][p].keys()):    
        outfile.write("\t"+p+"_"+d)

outfile.write("\n")

#model
for m in sorted(results_dict.keys()):
    #fine-tune
    for f in sorted(results_dict[m].keys()):
        outfile.write(m+'_'+f)
        #probing_type
        for p in sorted(results_dict[m][f].keys()):
            #data set
            for d in sorted(results_dict[m][f][p].keys()):
                outfile.write("\t"+str(round(float(results_dict[m][f][p][d]), 3)))

        outfile.write("\n")




