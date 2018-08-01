import pandas as pd
import conllu
import json

def get_ud_corpus_data():
  '''
  extracts the underlying UD corpus data that is stored in conllu format.
  Returns a dictionary where the keys are the split and the values are dictionaries
  where the keys are the sentenceId
  '''
  extracted_data_by_split = {}
  sentId2text = {}
  for split in ['train', 'dev', 'test']:
    extracted_data_by_split[split] = {}
    data = open("UD_English-EWT-r1.2/en-ud-%s.conllu" % (split)).readlines()
    data = "".join(data)
    data = conllu.parse(data)
    sent_count = 0
    for sent in data:
      extracted_data_by_split[split][sent_count] = sent
      sentId2text[(split, sent_count)] = " ".join([item['form'] for item in sent] )
      sent_count += 1

  return extracted_data_by_split, sentId2text

def convert_spr(ud_corpus):    

    sent_id2pred_arg_pairs = {}
    sent_id2targets = {}
    df = pd.read_csv("protoroles_eng_ud1.2_11082016.tsv", sep='\t', header=0)
    for df_idx, row in df.iterrows():
        if row['Applicable'] == 'no':
            continue
        id_pair = row['Sentence.ID'].split()
        assert len(id_pair) == 2
        split = id_pair[0].split(".con")[0].split("-")[-1]
        sent_id = id_pair[1]
        sent_text = ud_corpus[1][split,int(sent_id)-1]
        #Dataset    Is.Pilot    Passes.Filters  Protocol    Split   Annotator.ID    Sentence.ID Pred.Token  Pred.Lemma  Gram.Func   Arg.Phrase  Arg.Tokens.Begin    Arg.Tokens.End  Property    Response    Applicable  Sent.Grammatical
        #assert " ".join(sent_text.split()[row['Arg.Tokens.Begin']: row['Arg.Tokens.End']+ 1]).lower() == row['Arg.Phrase'].lower(), " ".join(sent_text.split()[row['Arg.Tokens.Begin']: row['Arg.Tokens.End']+ 1]) + "\t||\t" + row['Arg.Phrase'] 
        if (split, sent_id) not in sent_id2targets:
            sent_id2targets[(split, sent_id)] = {"targets": [],
                                                 "info": {
                                                    "source": "SPR2",
                                                    "sent-id": sent_id,
                                                    "split": split,
                                                    "grammatical" : row['Sent.Grammatical']
                                                     }
                                                 }
            sent_id2pred_arg_pairs[(split, sent_id)] = {}
        #span2 is argument, span1 is predicate
        span1 = (row['Pred.Token'], row['Pred.Token']+1)
        span2 = (row['Arg.Tokens.Begin'], row['Arg.Tokens.End']+1)
        if (span1, span2) not in sent_id2pred_arg_pairs[(split, sent_id)]:
            sent_id2pred_arg_pairs[(split, sent_id)][(span1, span2)] =  {
                "span2" : list(span2),
                "span1" : list(span1),
                "label"  : {},
                "info"      : {
                                "span2_txt" : row['Arg.Phrase'],
                                "span1_text": sent_text.split()[row['Pred.Token']],
                                "is_pilot"  : row['Is.Pilot'],
                                #"applicable": [], #row['Applicable'],
                                "pred_lemma": row['Pred.Lemma']
                              }
            }
        if row['Property'] not in sent_id2pred_arg_pairs[(split, sent_id)][(span1, span2)]["label"]:
            sent_id2pred_arg_pairs[(split, sent_id)][(span1, span2)]["label"][row['Property']] = []
        sent_id2pred_arg_pairs[(split, sent_id)][(span1, span2)]["label"][row['Property']].append(row['Response'])
        #sent_id2pred_arg_pairs[(split, sent_id)][(span1, span2)]["info"]["applicable"].append(row['Applicable'])
        #sent_id2targets[(split, sent_id)]["targets"].append(target)

    outfiles = {}
    for key in sent_id2targets:
        val = sent_id2targets[key]
        val["text"] = ud_corpus[1][key[0],int(key[1])-1]
        val["info"]["split"] = key[0]
        val["info"]["sent_id"] = key[1]
        for span_pair in sent_id2pred_arg_pairs[key]:
            labels = sent_id2pred_arg_pairs[key][span_pair]["label"]
            sent_id2pred_arg_pairs[key][span_pair]["label"] = [key for key,val in labels.items() if sum(val) / float(len(val)) >= 4.0 ]
            val["targets"].append(sent_id2pred_arg_pairs[key][span_pair])
            #json_objs.append(val)

            #for obj in json_objs:
        data_split = val['info']['split']
        if data_split not in outfiles:
            outfiles[data_split] = open("spr2_edge_probing_%s.json" % (data_split), 'w')
        json.dump(val, outfiles[data_split])
        outfiles[data_split].write('\n')

def main():
    ud_corpus = get_ud_corpus_data()
    convert_spr(ud_corpus)

if __name__ == '__main__':
    main()
