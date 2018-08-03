#!/usr/bin/env python

# Helper script to generate edge-probing json file recasted from Conll2003 dataset.
# Generated files are named CoNLL-2003_train.json, CoNLL-2003_dev.json, and CoNLL-2003_test.json.
# Files with the same names will be overwritten.
#
# Usage:
#  python conll2003_process.py -env /path/to/conll2003/environment

from allennlp.data.dataset_readers import conll2003 
from allennlp.data.dataset_readers.dataset_utils import span_utils
import json
import sys
import argparse

def read_conll2003(filepath):
    reader = conll2003.Conll2003DatasetReader()
    instance_list = []
    for instance in reader._read(filepath):
        instance_list.append(instance)
    return instance_list

def convert_instance_to_dict(instance):
    instance_dict = {}
    instance_dict['text'] = ' '.join( word for word in [str(token) for token in instance.fields['tokens'].tokens])
    instance_dict["targets"] = []
    labels_list = instance.fields['tags'].labels
    targets_list = span_utils.bio_tags_to_spans(labels_list) #list of ('TAG', (int, int))
    for BIO_tag, span in targets_list:
        instance_dict["targets"].append({"span1": [span[0], span[1]+1], "label": BIO_tag, "info": {}})
    instance_dict["info"] = {"source": "conll2003"}
    return instance_dict

def generate_conll2003_json_for_EPData(instance_list, split):
    data = [convert_instance_to_dict(datum) for datum in instance_list]
    with open('CoNLL-2003_' + split + '.json', 'w') as outfile:
        for datum in data:
            json.dump(datum, outfile)
            outfile.write("\n")

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', dest='data_env', type=str, required=True,
                        default="/home/bchen6_swarthmore_edu",
                        help="Path environment hosting the  CoNLL 2003 data.")
    args = parser.parse_args(args)

    dev_filepath = args.data_env + '/CoNLL-2003/eng.testa'
    test_filepath = args.data_env + '/CoNLL-2003/eng.testb'
    train_filepath = args.data_env + '/CoNLL-2003/eng.train'
    
    dev_data = read_conll2003(dev_filepath) #list of Instance objects
    test_data = read_conll2003(test_filepath)
    train_data = read_conll2003(train_filepath)    

    generate_conll2003_json_for_EPData(dev_data, "dev")
    generate_conll2003_json_for_EPData(test_data, "test")
    generate_conll2003_json_for_EPData(train_data, "train")

    print("Done.")


if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)
