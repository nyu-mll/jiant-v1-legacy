# Edge Probing Data

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

Following the final expt scripts for the other subgroups, just push a commit to this branch with your documentation & scripts, and we'll merge everything as a single PR at the end of the week.


## Semantic Role Labeling (TODO: Ian)

Tasks: `edges-srl-conll2005`, `edges-srl-conll2012`

Do as I say, not as I do... Will need to chase down Ian to get the code checked in before publication. Fortunately, it's easy for Ian to chase down Ian.

Original data prepared by following instructions from [He et al. 2017](https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf), as given here: [https://github.com/luheng/deep_srl#conll-data](https://github.com/luheng/deep_srl#conll-data)

Processed from the DeepSRL format into protocol buffers on Ian's Google workstation, then converted into edge probing JSON format via a Colaboratory notebook. TODO(Ian) to reconstruct this directly from the original data and check scripts in here.

## Nonced data (TODO: Najoung)

Lorem ipsum...

## Semantic Proto Roles v2 (Adam)

Tasks: ` `

To get the original data run `bash get_spr2_data.sh`.
To convert the data, run `python conver-spr2.py`

* This requires the python package called `conllu` *

## Definite Pronoun Resolution (Adam)

Tasks: ` `

To get the original data, run `bash get_dpr_data.sh`.
To convert the data, run `python convert-dpr.py`

## Winobias and Winogender (TODO: Patrick)

Lorem ipsum...

## OntoNotes (TODO: Patrick)

Lorem ipsum...

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

