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

## Semantic Proto Roles v2 (TODO: Adam)

Lorem ipsum...

## Definite Pronoun Resolution (TODO: Adam)

Lorem ipsum...

## Winobias and Winogender (TODO: Patrick)

Lorem ipsum...

## OntoNotes (TODO: Patrick)

Lorem ipsum...

## CoNLL 2003 NER (TODO: Berlin)

Lorem ipsum...

## Penn Treebank Constituent Labeling (TODO: Berlin)

Lorem ipsum...

## Universal Dependencies (TODO: Tom)

Lorem ipsum...

## CCG Tagging & Parsing (TODO: Tom)

Lorem ipsum...

