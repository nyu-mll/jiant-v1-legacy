"""
Task definitions for language modeling tasks set up to use for unsupervised parsing encoders.
Long term dependencies for language modeling: sentences concatenated together seperated by
<EOS> token.
"""
import math
import os
from typing import Iterable, Type

# Fields for instance processing
from allennlp.data import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.training.metrics import Average

from ..utils.data_loaders import load_tsv
from .lm import LanguageModelingTask
from .registry import register_task
from .tasks import sentence_to_text_field

from .tasks import (
    UNK_TOK_ALLENNLP,
    UNK_TOK_ATOMIC,
    SequenceGenerationTask,
    atomic_tokenize,
    sentence_to_text_field,
)



class LanguageModelingParsingTask(LanguageModelingTask):
    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """Process a language modeling split by indexing and creating fields.
        Args:
            split: (list) a single list of sentences
            indexers: (Indexer object) indexer to index input words
        """

        def _make_instance(sent):
            """ Forward targs adds <s> as a target for input </s>
            and bwd targs adds </s> as a target for input <s>
            to avoid issues with needing to strip extra tokens
            in the input for each direction """
            d = {}
            d["input"] = sentence_to_text_field(sent[:-1], indexers)
            d["targs"] = sentence_to_text_field(sent[1:], indexers)
            d["targs_b"] = sentence_to_text_field([sent[-1]] + sent[:-2], indexers)
            return Instance(d)

        for sent in split:
            yield _make_instance(sent)


@register_task("wsj", rel_path="WSJ/")
class WSJLanguageModelling(LanguageModelingParsingTask):
    """ Language modeling on a PTB dataset
    See base class: LanguageModelingTask
    """

    def get_data_iter(self, path):
        """ Rather than return a whole list of examples, stream them """
        nonatomics_toks = [UNK_TOK_ALLENNLP, "<unk>"]
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                # WikiText103 preprocesses unknowns as '<unk>'
                # which gets tokenized as '@', '@', 'UNKNOWN', ...
                # We replace to avoid that
                sent = atomic_tokenize(
                    toks,
                    UNK_TOK_ATOMIC,
                    nonatomics_toks,
                    self.max_seq_len,
                    tokenizer_name=self._tokenizer_name,
                )
                # we also filtering out headers (artifact of the data)
                # which are processed to have multiple = signs
                if sent.count("=") >= 2 or len(toks) < self.min_seq_len + 2:
                    continue
                yield sent


@register_task("toronto_lm", rel_path="toronto/")
class TorontoLanguageModelling(LanguageModelingParsingTask):
    """ Language modeling on the Toronto Books dataset
    See base class: LanguageModelingTask
    """

    def get_data_iter(self, path):
        """Load data file, tokenize text and concat sentences to create long term dependencies.
        Args:
            path: (str) data file path
        """
        seq_len = self.max_seq_len
        tokens = []
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                toks_v = toks.split()
                toks = toks.split() + ["<EOS>"]
                tokens += toks
            for i in range(0, len(tokens), seq_len):
                yield tokens[i : i + seq_len]


@register_task("egw_lm", rel_path="egw_corpus/")
class EnglishgigawordLanguageModeling(LanguageModelingParsingTask):
    """ Language modeling on the English Gigaword dataset
    See base class: LanguageModelingTask
    """

    def get_data_iter(self, path):
        """Load data file, tokenize text and concat sentences to create long term dependencies.
        Args:
            path: (str) data file path
        """
        seq_len = self.max_seq_len
        tokens = []
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                toks_v = toks.split()
                toks = toks.split() + ["<EOS>"]
                tokens += toks
            for i in range(0, len(tokens), seq_len):
                yield tokens[i : i + seq_len]


@register_task("mnli_lm", rel_path="MNLI/")
class MNLILanguageModeling(LanguageModelingParsingTask):
    """ Language modeling on the MNLI dataset
    See base class: LanguageModelingTask
    """

    def __init__(self, path, max_seq_len, name="mnli_lm", **kw):
        """Init class
        Args:
            path: (str) path that the data files are stored
            max_seq_len: (int) maximum length of one sequence
            name: (str) task name
        """
        super().__init__(path, max_seq_len, name, **kw)
        self.scorer1 = Average()
        self.scorer2 = None
        self.val_metric = "%s_perplexity" % self.name
        self.val_metric_decreases = True
        self.max_seq_len = max_seq_len
        self.min_seq_len = 0
        self.target_indexer = {"words": SingleIdTokenIndexer(namespace="tokens")}
        self.files_by_split = {
            "train": os.path.join(path, "train.tsv"),
            "val": os.path.join(path, "dev_matched.tsv"),
            "test": os.path.join(path, "test_matched.tsv"),
        }

    def get_data_iter(self, path):
        """
        Load data file (combine the entailment and contradiction sentence), tokenize text
         and concat sentences to create long term dependencies.
        Args:
            path: (str) data file path
        """
        seq_len = self.max_seq_len
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        data = load_tsv(
            os.path.join(path),
            1000,
            skip_rows=1,
            s1_idx=8,
            s2_idx=9,
            targ_idx=11,
            targ_map=targ_map,
        )
        tokens = []
        for x, y in zip(data[0], data[1]):
            tokens += x[1:-1] + ["<EOS>"] + y[1:-1] + ["<EOS>"]
        for i in range(0, len(tokens), seq_len):
            yield tokens[i : i + seq_len]
