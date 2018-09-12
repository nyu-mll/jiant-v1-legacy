#!/usr/bin/env python

# Script to convert event factuality data (the "It Happened v2" dataset) from
# conll format as provided by Rachel Rudinger and used in 
# https://arxiv.org/pdf/1804.02472.pdf. Output is JSON in edge probing format.
#
# This script requires the 'conllu' library; get it with:
#   pip install conllu
#
# Usage:
#   ./convert-ih2-rudinger.py -i /path/to/UDS_IH2_unified/*.conll \
#       -o /path/to/edges/data/ih2
#
# Input should be CoNLL format where the third column contains the filtered 
# regression score as used in the above paper. For example:
#
#  0	From	_	_	from	ADP	IN	_	2	case	_	_
#  1	the	_	_	the	DET	DT	Definite=Def|PronType=Art	2	det	_	_
#  2	AP	_	_	AP	PROPN	NNP	Number=Sing	3	nmod	_	_
#  3	comes	3.0	3.0	come	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	root	_	_
#  4	this	_	_	this	DET	DT	Number=Sing|PronType=Dem	5	det	_	_
#  5	story	_	_	story	NOUN	NN	Number=Sing	3	nsubj	_	_
#  6	:	_	_	:	PUNCT	:	_	3	punct	_	_

import sys
import os
import json
import collections
import argparse
from typing import Iterable, Dict

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import utils
import numpy as np
import pandas as pd
from tqdm import tqdm

import conllu

def make_record(conll_sentence):
    record = {}
    record['text'] = " ".join(t['form'] for t in conll_sentence)
    
    targets = []
    for token in conll_sentence:
        if token['filtered_rating'] == '_':
            continue
        score = float(token['filtered_rating'])
        span = (token['id'], token['id'] + 1)
        targets.append({'span1': span, 'label': score})
    record['targets'] = targets
    return record

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='inputs', type=str, nargs="+",
                        help="Input files (conll) for it-happened splits.")
    parser.add_argument('-o', dest='output_dir', type=str, required=True,
                        help="Output directory.")
    args = parser.parse_args(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    pd.options.display.float_format = '{:.2f}'.format
    cols = ('id', 'form', 'filtered_rating')
    for fname in args.inputs:
        log.info("Converting %s", fname)
        with open(fname) as fd:
            data = conllu.parse(fd.read(), fields=cols)
        records = (make_record(s) for s in tqdm(data))
        stats = utils.EdgeProbingDatasetStats()
        records = stats.passthrough(records)  # count records, tokens, etc.
        target_fname = os.path.join(args.output_dir, 
                                    os.path.basename(fname) + ".json")
        utils.write_json_data(target_fname, records)
        log.info("Wrote examples to %s", target_fname)
        log.info(stats.format())

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)

