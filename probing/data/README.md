# Edge Probing Datasets

This directory contains scripts to process the source datasets and generate the JSON data for edge probing tasks. Let `$JIANT_DATA_DIR` be the jiant data directory from the [main README](../../README.md), and make a subdirectory for the edge probing data:

```
mkdir $JIANT_DATA_DIR/edges
```

**TODO(all):** please document data-preparation scripts in this file! Add one or more sections for the tasks that you processed, describing how to produce the JSON data from the standard / distributed versions of the base datasets. Be sure to include:

- Links to the public / LDC versions of the raw data (so that an external user
  can repeat this from scratch).
- Scripts and instructions to run them on the raw data to produce the edge
  probing JSON files.
- A reference for the dataset that includes split information (# of
  examples in train/dev/test, etc.)
- Description of which splits you used, if there's any choices we made here
  (e.g. for UD treebank, that we used the English Web Treebank section, or for OntoNotes which test set we're using and why).

Let's put all the scripts in this folder (`jiant/probing/data`) so we don't clutter up the main `src/` directory.

The resulting JSON has one example per line, with the following structure (line breaks added for clarity).

**SRL example**
```js
// span1 is predicate, span2 is argument
{
  “text”: “Ian ate strawberry ice cream”,
  “targets”: [
    { “span1”: [1,2], “span2”: [0,1], “label”: “A0” },
    { “span1”: [1,2], “span2”: [2,5], “label”: “A1” }
  ],
  “info”: { “source”: “PropBank”, ... }
}
```

**Constituents example**
```js
// span2 is unused
{
  “text”: “Ian ate strawberry ice cream”,
  “targets”: [
    { “span1”: [0,1], “label”: “NNP” },
    { “span1”: [1,2], “label”: “VBD” },
    ...
    { “span1”: [2,5], “label”: “NP” }
    { “span1”: [1,5], “label”: “VP” }
    { “span1”: [0,5], “label”: “S” }
  ]
  “info”: { “source”: “PTB”, ... }
}
```

**Semantic Proto-roles (SPR) example**
```js
// span1 is predicate, span2 is argument
// label is a list of attributes (multilabel)
{
  'text': "The main reason is Google is more accessible to the global community and you can rest assured that it 's not going to go away ."
  'targets': [
    {
      'span1': [3, 4], 'span2': [0, 3],
      'label': ['existed_after', 'existed_before', 'existed_during',
                'instigation', 'was_used'],
      'info': { ... }
    },
    ...
  ]
  'info': {'source': 'SPR2', ... },
}
```

## Labels and Retokenization

For each of the tasks below, we need to perform two more preprocessing steps.

First, extract the set of available labels:
```
export TASK_DIR="$JIANT_DATA_DIR/edges/<task>"
python jiant/probing/get_edge_data_labels.py -o $TASK_DIR/labels.txt \
    -i $TASK_DIR/*.json -s
```

Second, make retokenized versions for MosesTokenizer and for the OpenAI BPE model:
```
python jiant/probing/retokenize_edge_data.py $TASK_DIR/*.json
python jiant/probing/retokenize_edge_data.openai.py $TASK_DIR/*.json
```
This will make retokenized versions alongside the original files.

## OntoNotes
Tasks:
- Constituents / POS: `edges-constituent-ontonotes`, `edges-nonterminal-ontonotes`,
  `edges-pos-ontonotes`
- Entities: `edges-ner-ontonotes`
- SRL: `edges-srl-ontonotes`
- Coreference: `edges-coref-ontonotes-conll`

### Getting OntoNotes Data
Follow the instructions at http://cemantix.org/data/ontonotes.html; you should end up with a folder named `conll-formatted-ontonotes-5.0/`.

If you're working on the JSALT cloud project, you can also download this directly from `gs://jsalt-data/ontonotes`.

### Extracting Data
To extract all OntoNotes tasks, run:
```
python extract_ontonotes_all.py --ontonotes /path/to/conll-formatted-ontonotes-5.0 \
  --tasks const coref ner srl \
  --splits train development test conll-2012-test \
  -o $JIANT_DATA_DIR/edges/ontonotes
```
This will write a number of JSON files, one for each split for each task, with names `{task}/{split}.json`.

### Splitting Constituent Data

The constituent data from the script above includes both preterminal (POS tag) and nonterminal (constituent) examples. We can split these into the `edges-nonterminal-ontonotes` and `edges-pos-ontonotes` tasks by running:
```
python jiant/probing/split_constituent_data.py $JIANT_DATA_DIR/edges/ontonotes/const/*.json
```
This will create `*.pos.json` and `*.nonterminal.json` versions of each input file.

## Semantic Role Labeling (TODO: Ian)

Tasks: `edges-srl-conll2005`, `edges-srl-conll2012`

Do as I say, not as I do... Will need to chase down Ian to get the code checked in before publication. Fortunately, it's easy for Ian to chase down Ian.

Original data prepared by following instructions from [He et al. 2017](https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf), as given here: [https://github.com/luheng/deep_srl#conll-data](https://github.com/luheng/deep_srl#conll-data)

Processed from the DeepSRL format into protocol buffers on Ian's Google workstation, then converted into edge probing JSON format via a Colaboratory notebook. TODO(Ian) to reconstruct this directly from the original data and check scripts in here.


## Semantic Proto Roles (SPR)

### SPR1
Tasks: `edges-spr1`

The version of SPR1 distributed on [decomp.io](http://decomp.io/) is difficult to work with directly, because it requires joining with both the Penn Treebank and the PropBank SRL annotations. If you have access to the Penn Treebank ([LDC99T42](https://catalog.ldc.upenn.edu/ldc99t42)), contact Rachel Rudinger or Ian Tenney for a processed copy of the data.

From Rachel's JSON format, you can use a script in this directory to convert to edge probing format:

```
./convert-spr1-rudinger.py -i /path/to/spr1/*.json \
    -o $JIANT_DATA_DIR/edges/spr1
```

You should get files named `spr1.{split}.json` where `split = {train, dev, test}`.

### SPR2
Tasks: `edges-spr2`

Run:
```
pip install conllu
./get_spr2_data.sh $JIANT_DATA_DIR/edges/spr2
```

This downloads both the UD treebank and the annotations and performs a join. See the `get_spr2_data.sh` script for more info. The `conllu` package is required to process the Universal Dependencies source data.


## Definite Pronoun Resolution (DPR)

Tasks: `edges-dpr`

Run:
```
./get_dpr_data.sh $JIANT_DATA_DIR/edges/dpr
```

## Winobias and Winogender (TODO: Patrick)

### WinoBias:

Download the data from [here](https://github.com/uclanlp/corefBias/tree/master/WinoBias/wino/data) (not the conll versions), then run `python extract_winobias.py OUTPUT_FILE_NAME *.{dev, test}`.

### WinoGender:

Download all the data from [here](https://github.com/rudinger/winogender-schemas/blob/master/data/all_sentences.tsv)
and then run `python extract_windogender.py OUTPUT_FILE_NAME all_sentences.tsv`.

## CoNLL 2003 NER

### Getting CoNLL 2003 NER Data

To download data, see [this](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003).  After successfully downloading `CoNLL-2003` to `/path/to/data`, you will see the following files in `CoNLL-2003`: `eng.testa`, `eng.testa.openNLP`, `eng.testb`, `eng.testb.openNLP`, `eng.train`, and `eng.train.openNLP`.  For train/dev/test split, we followed [standard practice](https://aclweb.org/aclwiki/CONLL-2003_(State_of_the_art)).

### Generating Edge Probing Data

Run `python conll2003_process.py -env /path/to/data` (there are python libraries that may need to be downloaded).  After a successful run, you will see files named `CoNLL-2003_dev.json`, `CoNLL-2003_test.json `, and `CoNLL-2003_train.json` in the current directory.  Files in the current directory with the same name as the three files generated will be overwritten.

## Penn Treebank Constituent Labeling

### Getting the Full PTB Dataset and Generating Edge Probing Data

Follow [this](https://www.ldc.upenn.edu/language-resources/data/obtaining) to obtain the full PTB dataset. After downloading/unzipping `ptb` directory to `/path/to/data`, run `python ptb_process -env /path/to/data` to generate `ptb_train.json`, `ptb_dev.json`, `ptb_dev.full.json`, and `ptb_test.json` (there are python libraries that may need to be downloaded).  Files in the current directory with the same name as the three files generated will be overwritten.

### Split of Penn Treebank (PTB) Dataset

For choice of train/dev/test, we followed [Klein *et al*](http://ilpubs.stanford.edu:8091/~klein/unlexicalized-parsing.pdf).  Note that there is a discrepency in the size of our dev set (first 20 files in section 22 of WSJ section of the Penn treebank), which has 427 sentences compared to 393 as described in Klein *et al*.

## Universal Dependencies

Tasks: `edges-dep-labeling-ewt`

Run:
```
./get_ud_data.sh $JIANT_DATA_DIR/edges/dep_ewt
```

This downloads the UD treebank and converts the conllu format to the edge probing format.
You should now see files named `en_ewt-ud-dev.json`, `en_ewt-ud-test.json`, and `en_ewt-ud-train.json` in the directory where the original data files were downloaded. Here is an example of one of the entries in the resulting file:

```
{
  "text": "This only serves their purposes .", 
  "targets": [
    {"span1": [0, 1], "span2": [2, 3], "label": "nsubj"}, 
    {"span1": [1, 2], "span2": [2, 3], "label": "advmod"}, 
    {"span1": [2, 3], "span2": [2, 3], "label": "root"}, 
    {"span1": [3, 4], "span2": [4, 5], "label": "nmod:poss"}, 
    {"span1": [4, 5], "span2": [2, 3], "label": "obj"}, 
    {"span1": [5, 6], "span2": [2, 3], "label": "punct"}
  ], 
  "info": {"source": "UD_English-EWT"}
}
```

Each element of `targets` represents one dependency arc. For example, the first element shows that there is an arc between the span going from index 0 to index 1 (i.e. the word "This") and the span going from index 2 to index 3 (i.e. the word "serves"), and that dependency arc has the label "nsubj." The task is to provide the correct labels for all of these dependency arcs.


## CCG Tagging

Download CCGBank from the [Linguistic Data Consortium](https://catalog.ldc.upenn.edu/LDC2005T13). 

Move the script `ccg_proc_tag.py` into the directory `ccgbank_1_1/data/AUTO` within the CCGBank download. Then, in that directory, run:
```
python ccg_proc_tag.py
```

This will create the files `ccg.tag.dev.json`, `ccg.tag.test.json`, and `ccg.tag.train.json`. Here is an example entry from one of these files:
```
{
  "text": "The play is filled with intrigue , dishonesty and injustice .", 
  "targets": [
    {"span1": [0, 1], "label": "NP[nb]/N"}, 
    {"span1": [1, 2], "label": "N"}, 
    {"span1": [2, 3], "label": "(S[dcl]\\NP)/(S[pss]\\NP)"}, 
    {"span1": [3, 4], "label": "(S[pss]\\NP)/PP"}, 
    {"span1": [4, 5], "label": "PP/NP"}, 
    {"span1": [5, 6], "label": "N"}, 
    {"span1": [6, 7], "label": ","}, 
    {"span1": [7, 8], "label": "N"}, 
    {"span1": [8, 9], "label": "conj"}, 
    {"span1": [9, 10], "label": "N"}, 
    {"span1": [10, 11], "label": "."}
  ], 
  "info": {"source": "ccgbank"}
}
```

Each span is a single word labeled with its CCG supertag.
