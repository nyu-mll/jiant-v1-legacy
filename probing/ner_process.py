import json

filepath = "/home/bchen6_swarthmore_edu/recast_ner_data.json"

with open(filepath, 'r') as f:
    data = json.load(f)

def find_span(phrase, sentence):
    split_sentence, split_phrase = sentence.split(), phrase.split()
    span_start = -1
    for i in range(len(split_sentence)- len(split_phrase) + 1):
        found = True
        for j in range(len(split_phrase)):
            if split_sentence[i + j] != split_phrase[j]:
                found = False 
        if found:
            span_start = i
            break
    if span_start == -1:
        print("Bad parsing.  Discounting target \"" + phrase + "\"")
        #raise RuntimeError("Span not found.")
    return [span_start, span_start + len(split_phrase)]

train_data, test_data = [], []

def json_to_json(split):
    interim_dict = {}
    for datum in data:
        if datum["split"] != split or datum["binary-label"] == False:
            continue
        text = datum["context"]
        if text not in interim_dict:
            interim_dict[text] = {'text': text, "targets": [], "info": {"sent-id": datum["sent-id"], "source": "NER"}}
        phrase, label = datum["hypothesis"].replace(" is an ", " is a ").split(" is a ")
        span1 = find_span(phrase, text)
        if span1[0] == -1:
            continue #Case when parsing is bad 
        interim_dict[text]["targets"].append({"label": label, \
                            "span1": span1, "info":{}})
    clean_data_list = interim_dict.values()
    with open('ner_' + split + ".json", 'w') as outfile:
        for datum in clean_data_list:
            json.dump(datum, outfile)
            outfile.write("\n")



json_to_json("train")
json_to_json("test")


