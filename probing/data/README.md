# Edge Probing Datasets

This directory contains scripts to process the source datasets and generate the JSON data for edge probing tasks.

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

## OntoNotes
Tasks:
- Constituents / POS: `constituent-ontonotes`, `nonterminal-ontonotes`,
  `pos-ontonotes`
- Entities: `ner-ontonotes`
- SRL: `srl-ontonotes`
- Coreference: `coref-ontonotes-conll`

### Getting OntoNotes Data
Follow the instructions at http://cemantix.org/data/ontonotes.html; you should end up with a folder named `conll-formatted-ontonotes-5.0/`.

If you're working on the JSALT cloud project, you can also download this directly from `gs://jsalt-data/ontonotes`.

### Extracting Data
To extract all OntoNotes tasks, run:
```
python extract_ontonotes_all.py --ontonotes /path/to/conll-formatted-ontonotes-5.0 \
  --tasks const coref ner srl \
  --splits train development test conll-2012-test \
  -o $OUTPUT_DIR
```
This will write a number of JSON files to `$OUTPUT_DIR`, one for each split for each task, with names `$OUTPUT_DIR/{task}.{split}.json`.


## Semantic Role Labeling (TODO: Ian)

Tasks: `edges-srl-conll2005`, `edges-srl-conll2012`

Do as I say, not as I do... Will need to chase down Ian to get the code checked in before publication. Fortunately, it's easy for Ian to chase down Ian.

Original data prepared by following instructions from [He et al. 2017](https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf), as given here: [https://github.com/luheng/deep_srl#conll-data](https://github.com/luheng/deep_srl#conll-data)

Processed from the DeepSRL format into protocol buffers on Ian's Google workstation, then converted into edge probing JSON format via a Colaboratory notebook. TODO(Ian) to reconstruct this directly from the original data and check scripts in here.

## Semantic Proto Roles (SPR)

Tasks: `spr1`, `spr2`

### SPR1

The version of SPR1 distributed on [decomp.io](http://decomp.io/) is difficult to work with directly, because it requires joining with both the Penn Treebank and the PropBank SRL annotations. If you have access to the Penn Treebank ([LDC99T42](https://catalog.ldc.upenn.edu/ldc99t42)), contact Rachel Rudinger or Ian Tenney for a processed copy of the data.

From Rachel's JSON format, you can use a script in this directory to convert to edge probing format:

```
./convert-spr1-rudinger.py -i /path/to/spr1/*.json \
    -o /path/to/probing/data/spr1/
```

You should get files named `spr1.{split}.json` where `split = {train, dev, test}`.

### SPR2

Run:
```
pip install conllu
./get_spr2_data.sh $JIANT_DATA_DIR/spr2
```

This downloads both the UD treebank and the annotations and performs a join. See the `get_spr2_data.sh` script for more info. The `conllu` package is required to process the Universal Dependencies source data.


## Definite Pronoun Resolution (DPR)

Tasks: `dpr`

To get the original data, run `bash get_dpr_data.sh`.
To convert the data, run `python convert-dpr.py`

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

## Universal Dependencies (TODO: Tom)

Lorem ipsum...

## CCG Tagging & Parsing (TODO: Tom)

Lorem ipsum...

