from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
import sys
import numpy as np
import json
 # from ptb_process import sent_to_dict

TYPE = sys.argv[3] # fill in with "ner"/coref/const/srl"

ontonotes = Ontonotes()
file_path = sys.argv[1] # e.g. test/train/development/conll-test: /nfs/jsalt/home/pitrack/ontonotes/ontonotes/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development
out_file = open(sys.argv[2], 'w+')
ontonotes_reader = ontonotes.dataset_iterator(file_path=sys.argv[1])

counter = []
num_span_pairs = 0
num_entities = 0
skip_counter = 0

def jsonify(spans, sentence, two_targets=False, frame=""):
    global num_span_pairs
    new_entry = {}
    new_entry["text"] = " ".join(sentence.words)
    def correct(span):
        global num_span_pairs
        num_span_pairs += 1
        # spans are of form (label, (begin, end)) (inclusive)
        # and get converted to (right-exclusive)
        #   {"span1": [begin, end], "label": label}
        if two_targets:
            return {"span1": [span[1][0], span[1][1] + 1],
                    "span2": [span[2][0], span[2][1] + 1],
                    "label": span[0]}
        else:
            return {"span1": [span[1][0], span[1][1] + 1],
                    "label": span[0]}
    new_entry["targets"] = [correct(span) for span in spans]
    new_entry["source"] = "{} {} {}".format(sentence.document_id, sentence.sentence_id, frame)
    return new_entry

def get_ners(sentence):
    global counter
    
    spans = bio_tags_to_spans(sentence.named_entities)
    counter.append(len(spans))
    return spans

def sent_to_dict(sentence):
    '''Function converting Tree object to dictionary compatible with common JSON format
     copied from ptb_process.py so it doesn't have dependencies
    '''
    form_function_discrepancies = ['ADV', 'NOM']
    grammatical_rule = ['DTV', 'LGS', 'PRD', 'PUT', 'SBJ', 'TPC', 'VOC']
    adverbials = ['BNF', 'DIR', 'EXT', 'LOC', 'MNR', 'PRP', 'TMP']
    miscellaneous = ['CLR', 'CLF', 'HLN', 'TTL']
    punctuations = ['-LRB-', '-RRB-', '-LCB-', '-RCB-', '-LSB-', '-RSB-']
    json_d = {}

    text = ""
    for word in sentence.flatten():
        text += word + " "
    json_d["text"] = text

    max_height = sentence.height()
    for i, leaf in enumerate(sentence.subtrees(lambda t: t.height() == 2)): #modify the leafs by adding their index in the sentence
        leaf[0] = (leaf[0], str(i))
    targets = []
    for index, subtree in enumerate(sentence.subtrees()):
        assoc_words = subtree.leaves()
        assoc_words = [(i, int(j)) for i, j in assoc_words]
        assoc_words.sort(key=lambda elem: elem[1])
        tmp_tag_list = subtree.label().replace('=', '-').replace('|', '-').split('-')
        label = tmp_tag_list[0]
        if tmp_tag_list[-1].isdigit(): #Getting rid of numbers at the end of each tag
            fxn_tgs = tmp_tag_list[1:-1]
        else:
            fxn_tgs = tmp_tag_list[1:]
        #Special cases:
        if len(tmp_tag_list) > 1 and tmp_tag_list[1] == 'S': #Case when we have 'PRP-S' or 'WP-S'
            label = tmp_tag_list[0] + '-' + tmp_tag_list[1]
            fxn_tgs = tmp_tag_list[2:-1] if tmp_tag_list[-1].isdigit() else tmp_tag_list[2:]
        if subtree.label() in punctuations: #Case when we have one of the strange punctions, such as round brackets
            label, fxn_tgs = subtree.label(), []
        targets.append({"span1":[int(assoc_words[0][1]), int(assoc_words[-1][1]) + 1], "label": label, \
                        "info": {"height": subtree.height() - 1, "depth": find_depth(sentence, subtree), \
                        "form_function_discrepancies": list(set(fxn_tgs).intersection(set(form_function_discrepancies))), \
                        "grammatical_rule": list(set(fxn_tgs).intersection(set(grammatical_rule))), \
                        "adverbials": list(set(fxn_tgs).intersection(set(adverbials))), \
                        "miscellaneous": list(set(fxn_tgs).intersection(set(miscellaneous)))}})
    json_d["targets"] = targets
    
    json_d["info"] = {"source": "PTB"}
    
    return json_d



def nltk_tree_to_spans(nltk_tree):
    # Input: nltk.tree
    # Output: List[(Str, (Int, Int))] of labelled spans
    # where the first element of the tuple is a string
    # and the second is a [begin, end) tuple specifying the span
    span_dict = sent_to_dict(nltk_tree)
    return span_dict
    

def get_consts(sentence):
    global counter, skip_counter
    try:
        span_dict = nltk_tree_to_spans(sentence.parse_tree)
    except Exception as e:
        skip_counter += 1
        return jsonify([], sentence)
    counter.append(len(span_dict['targets']))
    span_dict["source"] = "{} {}".format(sentence.document_id, sentence.sentence_id)
    return span_dict

def find_links(span_list):
  pairs = []
  for i, span_1 in enumerate(span_list):
    for span_2 in span_list[i+1:]:
        pairs.append((str(int(span_1[0] == span_2[0])),
                      span_1[1],
                      span_2[1]))
  return pairs


def get_corefs(sentence):
    global counter
    spans = find_links(list(sentence.coref_spans))
    counter.append(len(spans))
    return spans

def get_srls(sentence):
    global counter
    sentence_targets = []
    for frame, bio_tags in sentence.srl_frames:
        frame_targets = []
        spans = bio_tags_to_spans(bio_tags)
        head_span = None
        other_spans = []
        for (tag, indices) in spans:
            if tag == "V":
                head_span = indices
            else:
                other_spans.append((tag, indices))
        if head_span is None:
            print (frame, bio_tags)
        for span2_tag, span2 in other_spans:
            frame_targets.append((span2_tag, head_span, span2))
        sentence_targets.append(frame_targets)
        counter.append(len(other_spans) + 1)
    return sentence_targets
    

sent_counter = 0
for sentence in ontonotes_reader:
    sent_counter += 1
    # returns dict of spans, right-exclusive, STRING labeled with
    # named entity label
    if TYPE == "ner":
        spans = get_ners(sentence)
        out_file.write(json.dumps(jsonify(spans, sentence, two_targets=False)))
    elif TYPE == "const":
        spans_dict = get_consts(sentence)
        out_file.write(json.dumps(spans_dict))
    elif TYPE == "coref":
        spans = get_corefs(sentence)
        num_entities += len(sentence.coref_spans)
        out_file.write(json.dumps(jsonify(spans, sentence, two_targets=True)))
    elif TYPE == "srl":
        srls = get_srls(sentence)
        for frame in srls:
            out_file.write(json.dumps(jsonify(frame, sentence, two_targets=True, frame=frame)))
            out_file.write("\n")
    if TYPE != "srl":
        out_file.write("\n")

print ("num entities:{}".format(sum(counter)))
print ("some stats mn|std|md: {} {} {}".format(np.mean(counter), np.std(counter), np.median(counter)))
print ("hist: {}".format(np.histogram(counter)))
print ("num sents: {}".format(sent_counter))
print ("num_span_pairs: {}".format(num_span_pairs))
print ("also num ents: {}".format(num_entities))
print ("skipped: {}".format(skip_counter))
