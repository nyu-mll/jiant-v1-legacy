######################################################################
#  CliNER - documents.py                                             #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Build model for given training data.                     #
######################################################################



import string
import re
import nltk
import os

######################################################################
#  CliNER - tools.py                                                 #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: General purpose tools                                    #
######################################################################


import os
import sys
import errno
import string
import math
import re
import pickle
import numpy as np
from jiant.utils import retokenize
# from jiant.utils import moses_aligner


#############################################################
#  files
#############################################################

def map_files(files):
    """Maps a list of files to basename -> path."""
    output = {}
    for f in files: #pylint: disable=invalid-name
        basename = os.path.splitext(os.path.basename(f))[0]
        output[basename] = f
    return output


def mkpath(path):
    """Alias for mkdir -p."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

#############################################################
#  text pre-processing
#############################################################

def clean_text(text):
    return ''.join(map(lambda x: x if (x in string.printable) else '@', text))


def normalize_tokens(toks):
    # todo: normalize dosages (icluding 8mg -> mg)
    # replace number tokens
    def num_normalize(w):
        return '__num__' if re.search('\d', w) else w
    toks = list(map(num_normalize, toks))
    return toks


#############################################################
#  manipulating list-of-lists
#############################################################

def flatten(list_of_lists):
    '''
    flatten()
    Purpose: Given a list of lists, flatten one level deep
    @param list_of_lists. <list-of-lists> of objects.
    @return               <list> of objects (AKA flattened one level)
    >>> flatten([['a','b','c'],['d','e'],['f','g','h']])
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    '''
    return sum(list_of_lists, [])



def save_list_structure(list_of_lists):
    '''
    save_list_structure()
    Purpose: Given a list of lists, save way to recover structure from flattended
    @param list_of_lists. <list-of-lists> of objects.
    @return               <list> of indices, where each index refers to the
                                 beginning of a line in the orig list-of-lists
    >>> save_list_structure([['a','b','c'],['d','e'],['f','g','h']])
    [3, 5, 8]
    '''

    offsets = [ len(sublist) for sublist in list_of_lists ]
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i-1]

    return offsets




def reconstruct_list(flat_list, offsets):

    '''
    save_list_structure()
    Purpose: This undoes a list flattening. Uses value from save_list_structure()
    @param flat_list. <list> of objects
    @param offsets    <list> of indices, where each index refers to the
                                 beginning of a line in the orig list-of-lists
    @return           <list-of-lists> of objects (the original structure)
    >>> reconstruct_list(['a','b','c','d','e','f','g','h'], [3,5,8])
    [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h']]
    '''

    return [ flat_list[i:j] for i, j in zip([0] + offsets, offsets)]




#############################################################
#  serialization to disc
#############################################################

def load_pickled_obj(path_to_pickled_obj):
    data = None
    with open(path_to_pickled_obj, "rb") as f:
        data = f.read()
    return pickle.loads(data)


def pickle_dump(obj, path_to_obj):
    # NOTE: highest priority makes loading TRAINED models slow
    with open(path_to_obj, 'wb') as f:
        pickle.dump(obj, f, -1)



#############################################################
#  prose v nonprose
#############################################################


def is_prose_sentence(sentence):
    assert type(sentence) == type([]), 'is_prose_sentence() must take list arg'
    if sentence == []:
        return False
    #elif sentence[-1] == '.' or sentence[-1] == '?':
    elif sentence[-1] == '?':
        return True
    elif sentence[-1] == ':':
        return False
    elif len(sentence) <= 5:
        return False
    elif is_at_least_half_nonprose(sentence):
        return True
    else:
        return False



def is_at_least_half_nonprose(sentence):
    count = len(filter(is_prose_word, sentence))
    if count >= len(sentence)/2:
        return True
    else:
        return False



def is_prose_word(word):
    # Punctuation
    for punc in string.punctuation:
        if punc in word:
            return False
    # Digit
    if re.match('\d', word):
        return False
    # All uppercase
    if word == word.upper():
        return False
    # Else
    return True




def prose_partition(tokenized_sents, labels=None):
    prose_sents     = []
    nonprose_sents  = []
    prose_labels    = []
    nonprose_labels = []

    # partition the sents & labels into EITHER prose OR nonprose groups
    for i in range(len(tokenized_sents)):
        if is_prose_sentence(tokenized_sents[i]):
            prose_sents.append(tokenized_sents[i])
            if labels:
                prose_labels.append(labels[i])
        else:
            nonprose_sents.append(tokenized_sents[i])
            if labels:
                nonprose_labels.append(labels[i])

    # group data appropriately (note, labels might not be provided)
    if labels:
        prose    = (   prose_sents,    prose_labels)
        nonprose = (nonprose_sents, nonprose_labels)
    else:
        prose    = (   prose_sents, None)
        nonprose = (nonprose_sents, None)

    return prose, nonprose





def print_files(f, file_names):
    '''
    print_files()

    Pretty formatting for listing the training files in a 
    log.

    @param f.           An open file stream to write to.
    @param file_names.  A list of filename strings.
    '''
    COLUMNS = 4
    file_names = sorted(file_names)
    start = 0
    for row in range(int(math.ceil(float(len(file_names))/COLUMNS))):
        write(f, u'\t\t')
        for featname in file_names[start:start+COLUMNS]:
            write(f, '%-15s' % featname)
        write(f, u'\n')
        start += COLUMNS



# python2 needs to convert to unicdode, but thats default for python3
if sys.version_info.major == 2:
    tostr = unicode
else:
    tostr = str

def write(f, s):
    f.write(tostr(s))



def print_vec(f, label, vec):
    '''
    print_vec()

    Pretty formatting for displaying a vector of numbers in a log.

    @param f.           An open file stream to write to.
    @param label.  A description of the numbers (e.g. "recall").
    @param vec.    A numpy array of the numbers to display.
    '''
    COLUMNS = 7
    start = 0
    write(f, '\t%-10s: ' % label)
    if type(vec) != type([]):
        vec = vec.tolist()
    for row in range(int(math.ceil(float(len(vec))/COLUMNS))):
        for featname in vec[start:start+COLUMNS]:
            write(f, '%7.3f' % featname)
        write(f, u'\n')
        start += COLUMNS

        
        
def print_str(f, label, names):

    '''
    print_str()
    Pretty formatting for displaying a list of strings in a log
    @param f.           An open file stream to write to.
    @param label.  A description of the numbers (e.g. "recall").
    @param names.  A list of strings.
    '''
    COLUMNS = 4
    start = 0
    for row in range(int(math.ceil(float(len(names))/COLUMNS))):
        if row == 0:
            write(f, '\t%-10s: ' % label)
        else:
            write(f, '\t%-10s  ' % '')

        for featname in names[start:start+COLUMNS]:
            write(f, '%-16s ' % featname)
            
        write(f, u'\n')
        start += COLUMNS



#############################################################
#  Quick-and-Dirty evaluation of performance
#############################################################


def compute_performance_stats(label, pred, ref):
    '''
    compute_stats()
    Compute the P, R, and F for a given model on some data.
    @param label.  A name for the data (e.g. "train" or "dev")
    @param pred.   A list of list of predicted labels.
    @param pred.   A list of list of true      labels.
    '''

    num_tags = max(set(sum( ref,[])) | set(sum(pred,[]))) +1
    # confusion matrix
    confusion = np.zeros( (num_tags,num_tags) )
    for tags,yseq in zip(pred,ref):
        for y,p in zip(yseq, tags):
            confusion[p,y] += 1

    # print confusion matrix
    conf_str = ''
    conf_str += '\n\n'
    conf_str += label + '\n'
    conf_str += ' '*6
    for i in range(num_tags):
        conf_str += '%4d ' % i
    conf_str += ' (gold)\n'
    for i in range(num_tags):
        conf_str +=  '%2d    ' % i
        for j in range(num_tags):
            conf_str += '%4d ' % confusion[i][j]
        conf_str += '\n'
    conf_str += '(pred)\n'
    conf_str += '\n\n'
    conf_str = conf_str
    #print conf_str

    precision = np.zeros(num_tags)
    recall    = np.zeros(num_tags)
    f1        = np.zeros(num_tags)

    for i in range(num_tags):
        correct    =     confusion[i,i]
        num_pred   = sum(confusion[i,:])
        num_actual = sum(confusion[:,i])

        p  = correct / (num_pred   + 1e-9)
        r  = correct / (num_actual + 1e-9)

        precision[i] = p
        recall[i]    = r
        f1[i]        = (2*p*r) / (p + r + 1e-9)

    scores = {}
    scores['precision'] = precision
    scores['recall'   ] = recall
    scores['f1'       ] = f1
    scores['conf'     ] = conf_str

    return scores

labels = { 'O':0,
           'B-problem':1, 'B-test':2, 'B-treatment':3,
           'I-problem':4, 'I-test':5, 'I-treatment':6,
         }

id2tag = { v:k for k,v in labels.items() }


id2tag = { v:k for k,v in labels.items() }


class Document:

    def __init__(self, txt, tokenizer_name, max_seq_len, con=None):
        # read data
        retVal = read_i2b2(txt, con, tokenizer_name)
        # Internal representation natural for i2b2 format
        self._tok_sents = retVal[0]
        self.max_seq_len = max_seq_len
        # Store token labels
        if con:
            self._tok_concepts = retVal[1]
            self._labels = tok_concepts_to_labels(self._tok_sents,
                                                  self._tok_concepts, txt, con)
        self._tok_sents = [item for sublist in self._tok_sents for item in sublist]
        self._labels = [item for sublist in self._labels for item in sublist]
        assert len(self._labels) == len(self._tok_sents)
        self._tok_sents, self._labels = preprocess_tagging(self._tok_sents, self._labels, tokenizer_name)
        # save filename
        self._filename = txt



    def getExtension(self):
        return 'con'


    def getTokenizedSentences(self):
        assert len(self._tok_sents) == len(self._labels)
        if self.max_seq_len > len(self._tok_sents):
            return self._tok_sents
        return self._tok_sents[:self.max_seq_len]


    def getTokenLabels(self):
        if self.max_seq_len > len(self._labels):
            return self._labels
        return self._labels[:self.max_seq_len]


    def conlist(self):
        assert len(self._tok_sents) == len(self._labels)
        return self._labels


    def write(self, pred_labels=None):
        """
        Purpose: Return the given concept label predictions in i2b2 format

        @param  pred_labels.     <list-of-lists> of predicted_labels
        @return                  <string> of i2b2-concept-file-formatted data
        """

        # Return value
        retStr = ''

        # If given labels to write, use them. Default to classifications
        if pred_labels != None:
            token_labels = pred_labels
        elif self._labels != None:
            token_labels = self._labels
        else:
            raise Exception('Cannot write concept file: must specify labels')

        concept_tuples = tok_labels_to_concepts(self._tok_sents, token_labels)

        # For each classification
        for classification in concept_tuples:

            # Ensure 'none' classifications are skipped
            if classification[0] == 'none':
                raise('Classification label "none" should never happen')

            concept = classification[0]
            lineno  = classification[1]
            start   = classification[2]
            end     = classification[3]

            # A list of words (corresponding line from the text file)
            text = self._tok_sents[lineno-1]

            #print("\n" + "-" * 80)
            #print("classification: ", classification)
            #print("lineno:         ", lineno)
            #print("start:          ", start)
            #print("end             ", end)
            #print("text:           ", text)
            #print('len(text):      ', len(text))
            #print("text[start]:    ", text[start])
            #print("concept:        ", concept)

            datum = text[start]
            for j in range(start, end):
                datum += " " + text[j+1]
            datum = datum.lower()

            #print('datum:          ', datum)

            # Line:TokenNumber of where the concept starts and ends
            idx1 = "%d:%d" % (lineno, start)
            idx2 = "%d:%d" % (lineno, end  )

            # Classification
            label = concept

            # Print format
            retStr +=  "c=\"%s\" %s %s||t=\"%s\"\n" % (datum, idx1, idx2, label)

        # return formatted data
        return retStr.strip()

def preprocess_tagging(text, current_tags, tokenizer_name):
    """
    Input: 
        text: A word delimited list of text.
        current_tags: A word delimited list of tags. 
    Output: 
        Tags a word delimited list of tokenized tags. 
    """
    if tokenizer_name == "":
        aligner_fn = lambda x: ("", x.split())
    else:
        aligner_fn = retokenize.get_aligner_fn(tokenizer_name)
    assert len(text) == len(current_tags)
    res_tags = []
    new_toks_tis_way = []
    for i in range(len(text)):
        token = text[i]
        import re
        _, new_toks = aligner_fn(token)
        for tok in new_toks:
            new_toks_tis_way.append(tok)
            res_tags.append(current_tags[i])
            # based on BERT-paper for wordpiece, we only keep the tag
            # for the first part of the word.
    assert len(new_toks_tis_way) == len(res_tags)
    return new_toks_tis_way, res_tags


def read_i2b2(txt, con, tokenizer_name):
    """
    read_i2b2()
x
    @param txt. A file path for the tokenized medical record
    @param con. A file path for the i2b2 annotated concepts for txt
    """
    tokenized_sents = []

    sent_tokenize = lambda text: text.split('\n')
    word_split = lambda text: text.split(' ')

    # Read in the medical text
    with open(txt) as f:
        # Original text file
        text = f.read().strip('\n')

        # tokenize
        sentences = sent_tokenize(text)
        for sentence in sentences:
            sent = clean_text(sentence.rstrip())

            # lowercase 
            sent = sent.lower()

            toks = word_split(sent)

            # normalize tokens
            normed_toks = normalize_tokens(toks)

            #for w in normed_toks:
            #    print(w)
            #print()

            tokenized_sents.append(toks)
     # If an accompanying concept file was specified, read it
    tok_concepts = []
    if con:
        with open(con) as f:
            for line in f.readlines():
                # Empty line
                if not line.strip():
                    continue

                # parse concept line
                concept_regex = '^c="(.*)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*)"$'
                match = re.search(concept_regex, line.strip())
                groups = match.groups()

                # retrieve regex info
                concept_text  =     groups[0]
                start_lineno  = int(groups[1])
                start_tok_ind = int(groups[2])
                end_lineno    = int(groups[3])
                end_tok_ind   = int(groups[4])
                concept_label =     groups[5]

                # pre-process text for error-check
                #matching_line = tokenized_sents[start_lineno-1]
                #matching_toks = matching_line[start_tok_ind:end_tok_ind+1]
                #matching_text = ' '.join(matching_toks).lower()
                #concept_text  = ' '.join(word_tokenize(concept_text))

                # error-check info
                assert start_lineno==end_lineno, 'concept must span single line'
                #assert concept_text==matching_text, 'something wrong with inds'

                # add the concept info
                tup = (concept_label, start_lineno, start_tok_ind, end_tok_ind, concept_text)
                tok_concepts.append(tup)

        # Safe guard against concept file having duplicate entries
        tok_concepts = list(set(tok_concepts))

        # Concept file does not guarantee ordering by line number
        tok_concepts = sorted(tok_concepts, key=lambda t:t[1:])

        # Ensure no overlapping concepts (that would be bad)
        # BIO-Tagigng 
        add_pairs_to_delete = []
        for i in range(len(tok_concepts)-1):
            c1 = tok_concepts[i]
            c2 = tok_concepts[i+1]
            if c1[1] == c2[1]:
                if c1[2] <= c2[2] and c2[2] <= c1[3]:
                    add_pairs_to_delete.append(i+1)
                    fname = os.path.basename(con)
                    error1='%s has overlapping entities on line %d'%(fname,c1[1])
                    error2="It can't be processed until you remove one"
                    error3='Deleting entity 2 from list'
                    error4='\tentity 1: c="%s" %d:%d %d:%d||t="%s"'%(' '.join(tokenized_sents[c1[1]-1][c1[2]:c1[3]+1]),
                                                                     c1[1], c1[2], c1[1], c1[3], c1[0])
                    error5='\tentity 2: c="%s" %d:%d %d:%d||t="%s"'%(' '.join(tokenized_sents[c2[1]-1][c2[2]:c2[3]+1]),
                                                                     c2[1], c2[2], c2[1], c2[3], c2[0])
                    error_msg = '\n\n%s\n%s\n\n%s\n\n%s\n%s\n' % (error1,error2,error3,error4,error5)
        for index in sorted(add_pairs_to_delete, reverse=True):
            del tok_concepts[index]
    #result_sents, result_concepts = preprocess_tagging(tokenized_sents, tok_concepts, tokenizer_name)
    # result_sents. result_concepts
    return tokenized_sents, tok_concepts




def tok_concepts_to_labels(tokenized_sents, tok_concepts, tok_file, con_file):
    # parallel to tokens
    labels = [ ['O' for tok in sent] for sent in tokenized_sents ]

    # fill each concept's tokens appropriately
    for concept in tok_concepts:
        label,lineno,start_tok,end_tok, span_str = concept
        labels[lineno-1][start_tok] = 'B-%s' % label
        for i in range(start_tok+1,end_tok+1):
            labels[lineno-1][i] = 'I-%s' % label

    # test it out
    '''
    for i in range(len(tokenized_sents)):
        assert len(tokenized_sents[i]) == len(labels[i])
        for tok,lab in zip(tokenized_sents[i],labels[i]):
            if lab != 'O': print( '\t',)
            print (lab, tok)
        print()
    exit()
    '''

    return labels




def tok_labels_to_concepts(tokenized_sents, tok_labels):

    '''
    for gold,sent in zip(tok_labels, tokenized_sents):
        print(gold)
        print(sent)
        print()
    '''

    # convert 'B-treatment' into ('B','treatment') and 'O' into ('O',None)
    def split_label(label):
        if label == 'O':
            iob,tag = 'O', None
        else:
            iob,tag = label.split('-')
        return iob, tag

    # preprocess predictions to "correct" starting Is into Bs
    corrected = []
    for lineno,labels in enumerate(tok_labels):
        corrected_line = []
        for i in range(len(labels)):
            #'''
            # is this a candidate for error?
            iob,tag = split_label(labels[i])
            if iob == 'I':
                # beginning of line has no previous
                if i == 0:
                    print( 'CORRECTING! A')
                    new_label = 'B' + labels[i][1:]
                else:
                    # ensure either its outside OR mismatch type
                    prev_iob,prev_tag = split_label(labels[i-1])
                    if prev_iob == 'O' or prev_tag != tag:
                        print( 'CORRECTING! B')
                        new_label = 'B' + labels[i][1:]
                    else:
                        new_label = labels[i]
            else:
                new_label = labels[i]
            #'''
            corrected_line.append(new_label)
        corrected.append( corrected_line )

    '''
    for i,(trow,crow) in enumerate(zip(tok_labels, corrected)):
        if trow != crow:
            for j,(t,c) in enumerate(zip(trow,crow)):
                if t != c:
                    print('lineno: ', i)
                    print (t, '\t', c)
                    print()
            print()
    exit()
    '''
    tok_labels = corrected

    concepts = []
    for i,labs in enumerate(tok_labels):

        N = len(labs)
        begins = [ j for j,lab in enumerate(labs) if (lab[0] == 'B') ]

        for start in begins:
            # "B-test"  -->  "-test"
            label = labs[start][1:]

            # get ending token index
            end = start
            while (end < N-1) and tok_labels[i][end+1].startswith('I') and tok_labels[i][end+1][1:] == label:
                end += 1

            # concept tuple
            concept_tuple = (label[1:], i+1, start, end)
            concepts.append(concept_tuple)

    '''
    # test it out
    for i in range(len(tokenized_sents)):
        assert len(tokenized_sents[i]) == len(tok_labels[i])
        for tok,lab in zip(tokenized_sents[i],tok_labels[i]):
            if lab != 'O': print( '\t',)
            print (lab, tok)
        print()
    exit()
    '''

    # test it out
    test_tok_labels = tok_concepts_to_labels(tokenized_sents, concepts)
    #'''
    for lineno,(test,gold,sent) in enumerate(zip(test_tok_labels, tok_labels, tokenized_sents)):
        for i,(a,b) in enumerate(zip(test,gold)):
            #'''
            if not ((a == b)or(a[0]=='B' and b[0]=='I' and a[1:]==b[1:])):
                print()
                print( 'lineno:    ', lineno)
                print()
                print( 'generated: ', test[i-3:i+4])
                print( 'predicted: ', gold[i-3:i+4])
                print( sent[i-3:i+4])
                print( 'a[0]:  ', a[0])
                print( 'b[0]:  ', b[0])
                print( 'a[1:]: ', a[1:])
                print( 'b[1:]: ', b[1:])
                print( 'a[1:] == b[a:]: ', a[1:] == b[1:])
                print()
            #'''
            assert (a == b) or (a[0]=='B' and b[0]=='I' and a[1:]==b[1:])
            i += 1
    #'''
    assert test_tok_labels == tok_labels

    return concepts



class DocumentException(Exception):
    pass




