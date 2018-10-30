import sys
import os

# Converts CCG bank data into JSON format for CCG supertagging

directories_train = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"] #train
directories_dev = ["00"] #dev
directories_test = ["23"] #test



fo = open("ccg.tag.train.json", "w")
for directory in directories_train:
    for filename in os.listdir(directory):
        fi = open(directory + "/" + filename)

        for line in fi:
            if "<" in line:
                sentence = []
                tags = []
                leaf_tags = []

                items = line.strip().split("<")
                word_index = 0
                for item in items:
                    parts = item.split()

                    if parts[0] == "L":
                        sentence.append(parts[4])
                        tags.append([parts[1], (word_index, word_index + 1)])
                        leaf_tags.append([parts[1], (word_index, word_index + 1)])
                        word_index += 1
                    elif parts[0] == "T":
                        #tags.append([parts[1], (parts[3][:-1],)]) #for parse
                        pass

                if len(sentence) == len(leaf_tags):
                    new_tags = []
                    expanded = False

                    while not expanded:
                        expanded = True
                        for index, tag in enumerate(tags):
                            if len(tag[-1]) == 2:
                                new_tags.append(tag)

                            elif tag[-1][0] == "1":
                                if len(tags[index + 1][-1]) == 2:
                                    new_tags.append([tag[0], tags[index + 1][1]])
                                expanded = False

                            elif tag[-1][0] == "2":
                                if len(tags[index + 1][-1]) == 2 and len(tags[index + 2][-1]) == 2:
                                    new_tags.append([tag[0], (tags[index + 1][1][0], tags[index + 1][1][1])])
                                expanded = False

                            else:
                                print("ERROR")
                                print(tag)

                        tags = new_tags[:]

                    mod_tags = []
                    for tag in tags:
                        mod_tags.append('{"span1": [' + str(tag[1][0]) + ", " + str(tag[1][1]) + '], "label": "' + tag[0].replace("\\", "\\\\") + '"}')


                    fo.write('{"text": "' + " ".join(sentence).replace("\\", "\\\\").replace('"', '\\"') + '", "targets": [' + ", ".join(mod_tags) + '], "info": {"source": "ccgbank"}}\n') 




fo = open("ccg.tag.dev.json", "w")
for directory in directories_dev:
    for filename in os.listdir(directory):
        fi = open(directory + "/" + filename)

        for line in fi:
            if "<" in line:
                sentence = []
                tags = []
                leaf_tags = []

                items = line.strip().split("<")
                word_index = 0
                for item in items:
                    parts = item.split()

                    if parts[0] == "L":
                        sentence.append(parts[4])
                        tags.append([parts[1], (word_index, word_index + 1)])
                        leaf_tags.append([parts[1], (word_index, word_index + 1)])
                        word_index += 1
                    elif parts[0] == "T":
                        #tags.append([parts[1], (parts[3][:-1],)]) #for parse
                        pass

                if len(sentence) == len(leaf_tags):
                    new_tags = []
                    expanded = False

                    while not expanded:
                        expanded = True
                        for index, tag in enumerate(tags):
                            if len(tag[-1]) == 2:
                                new_tags.append(tag)

                            elif tag[-1][0] == "1":
                                if len(tags[index + 1][-1]) == 2:
                                    new_tags.append([tag[0], tags[index + 1][1]])
                                expanded = False

                            elif tag[-1][0] == "2":
                                if len(tags[index + 1][-1]) == 2 and len(tags[index + 2][-1]) == 2:
                                    new_tags.append([tag[0], (tags[index + 1][1][0], tags[index + 1][1][1])])
                                expanded = False

                            else:
                                print("ERROR")
                                print(tag)

                        tags = new_tags[:]

                    mod_tags = []
                    for tag in tags:
                        mod_tags.append('{"span1": [' + str(tag[1][0]) + ", " + str(tag[1][1]) + '], "label": "' + tag[0].replace("\\", "\\\\") + '"}')


                    fo.write('{"text": "' + " ".join(sentence).replace("\\", "\\\\").replace('"', '\\"') + '", "targets": [' + ", ".join(mod_tags) + '], "info": {"source": "ccgbank"}}\n') 




fo = open("ccg.tag.test.json", "w")
for directory in directories_test:
    for filename in os.listdir(directory):
        fi = open(directory + "/" + filename)

        for line in fi:
            if "<" in line:
                sentence = []
                tags = []
                leaf_tags = []

                items = line.strip().split("<")
                word_index = 0
                for item in items:
                    parts = item.split()

                    if parts[0] == "L":
                        sentence.append(parts[4])
                        tags.append([parts[1], (word_index, word_index + 1)])
                        leaf_tags.append([parts[1], (word_index, word_index + 1)])
                        word_index += 1
                    elif parts[0] == "T":
                        #tags.append([parts[1], (parts[3][:-1],)]) #for parse
                        pass

                if len(sentence) == len(leaf_tags):
                    new_tags = []
                    expanded = False

                    while not expanded:
                        expanded = True
                        for index, tag in enumerate(tags):
                            if len(tag[-1]) == 2:
                                new_tags.append(tag)

                            elif tag[-1][0] == "1":
                                if len(tags[index + 1][-1]) == 2:
                                    new_tags.append([tag[0], tags[index + 1][1]])
                                expanded = False

                            elif tag[-1][0] == "2":
                                if len(tags[index + 1][-1]) == 2 and len(tags[index + 2][-1]) == 2:
                                    new_tags.append([tag[0], (tags[index + 1][1][0], tags[index + 1][1][1])])
                                expanded = False

                            else:
                                print("ERROR")
                                print(tag)

                        tags = new_tags[:]

                    mod_tags = []
                    for tag in tags:
                        mod_tags.append('{"span1": [' + str(tag[1][0]) + ", " + str(tag[1][1]) + '], "label": "' + tag[0].replace("\\", "\\\\") + '"}')


                    fo.write('{"text": "' + " ".join(sentence).replace("\\", "\\\\").replace('"', '\\"') + '", "targets": [' + ", ".join(mod_tags) + '], "info": {"source": "ccgbank"}}\n') 









