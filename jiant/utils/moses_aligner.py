import sys

from nltk.tokenize.moses import MosesTokenizer

TOK = MosesTokenizer()


fo = open("hey" + ".moses", "w")

def tokenize_moses(old_text, old_tags):
    """
    old_tags: string 
    old_toks: string
    """
    old_toks = old_text
    new_toks = TOK.tokenize(" ".join(old_text))
    tags = old_tags

    new_tags = []
    tag_counter = 0
    next_covered = 0
    for index, word in enumerate(new_toks):
        if next_covered > 0:
            next_covered -= 1
            continue


        processed = old_toks[tag_counter].replace("&", "&amp;").replace("'", "&apos;").replace(">", "&gt;").replace("\"", "&quot;")

        if word == old_toks[tag_counter].replace("&", "&amp;").replace("'", "&apos;").replace(">", "&gt;").replace("\"", "&quot;") or  \
        len(processed.split(word)) > 1:
            # sometimes they're a part
            if not (processed.split(word)[0] == "" and processed.split(word)[1] == ""):
                # kit is in there and it's not trivia,ly equal 
                new_tags.append(tags[tag_counter])
            else:
                new_tags.append(tags[tag_counter])
                tag_counter += 1
        elif word == old_toks[tag_counter + 1].replace("&", "&amp;").replace("'", "&apos;").replace(">", "&gt;").replace("\"", "&quot;"):
            new_tags.append(tags[tag_counter])
            tag_counter += 2
        elif word in old_toks[tag_counter + 1].replace("&", "&amp;").replace("'", "&apos;").replace(">", "&gt;").replace("\"", "&quot;"):
            if not (processed.split(word)[0] == "" and processed.split(word)[1] == ""):
                # kit is in there and it's not trivia,ly equal 
                new_tags.append(tags[tag_counter])
            else:
                new_tags.append(tags[tag_counter])
                tag_counter += 2
        else:
            for i in range(20):
                if word + "".join(new_toks[index + 1 : index + 1 + i + 1]) == old_toks[
                    tag_counter
                ].replace("&", "&amp;").replace("'", "&apos;").replace(">", "&gt;").replace("\"", "'&quot;"):
                    for k in range(i + 2):
                        new_tags.append(tags[tag_counter])
                    tag_counter += 1
                    next_covered = i + 1

            if next_covered == 0:
                import pdb; pdb.set_trace()
                return "", ""
                print("ERROR!", word, word + new_toks[index + 1], old_toks[tag_counter])
                print(" ".join(old_toks))
                print(" ".join(new_toks))
                print(" ")
    if len(new_tags) != len(new_toks):
        print("MISMATCH!!!")
    """
    fo.write(
        (" ".join(new_toks) + "\t" + " ".join(new_tags) + "\n")
        .replace("&amp;", "&")
        .replace("&apos;", "'")
    )

    """
    new_toks = " ".join(new_toks).replace("&amp;", "&").replace("&apos;", "'").split()
    new_tags = " ".join(new_tags).replace("&amp;", "&").replace("&apos;", "'").split()
    return new_toks, new_tags                                           
