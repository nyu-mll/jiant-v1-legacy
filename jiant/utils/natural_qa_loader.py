import collections
import json
import gzip

def candidates_iter(e):
  """Yield's the candidates that should not be skipped in an example."""
  for idx, c in enumerate(e["long_answer_candidates"]):
    if should_skip_context(e, idx):
      continue
    yield idx, c

def get_candidate_type_and_position(e, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "[NoLongAnswer]"
  else:
    return e["long_answer_candidates"][idx]["type_and_position"]


def get_text_span(example, span):
  """Returns the text in the example's document in the given token span."""
  token_positions = []
  tokens = []
  for i in range(span["start_token"], span["end_token"]):
    t = example["document_tokens"][i]
    if not t["html_token"]:
      token_positions.append(i)
      token = t["token"].replace(" ", "")
      tokens.append(token)
  return (token_positions, " ".join(tokens))


def get_candidate_text(e, idx):
  """Returns a text representation of the candidate at the given index."""
  # No candidate at this index.
  if idx < 0 or idx >= len(e["long_answer_candidates"]):
    return ([], "")

  # This returns an actual candidate.
  return get_text_span(e, e["long_answer_candidates"][idx])

def should_skip_context(e, idx):
  if not e["long_answer_candidates"][idx]["top_level"]:
    return True
  elif not get_candidate_text(e, idx)[1].strip():
    # Skip empty contexts.
    return True
  else:
    return False


def get_candidate_type(e, idx):
  """Returns the candidate's type: Table, Paragraph, List or Other."""
  c = e["long_answer_candidates"][idx]
  first_token = e["document_tokens"][c["start_token"]]["token"]
  if first_token == "<Table>":
    return "Table"
  elif first_token == "<P>":
    return "Paragraph"
  elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
    return "List"
  elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
    return "Other"
  else:
    tf.logging.warning("Unknoww candidate type found: %s", first_token)
    return "Other"

def token_to_char_offset(e, candidate_idx, token_idx):
  """Converts a token index to the char offset within the candidate."""
  c = e["long_answer_candidates"][candidate_idx]
  char_offset = 0
  for i in range(c["start_token"], token_idx):
    t = e["document_tokens"][i]
    if not t["html_token"]:
      token = t["token"].replace(" ", "")
      char_offset += len(token) + 1
  return char_offset

def add_candidate_types_and_positions(e):
    max_position = 384
    """Adds type and position info to each candidate in the document."""
    counts = collections.defaultdict(int)
    for idx, c in candidates_iter(e):
        context_type = get_candidate_type(e, idx)
        if counts[context_type] < max_position:
            counts[context_type] += 1
        c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])

def has_long_answer(a):
  return (a["long_answer"]["start_token"] >= 0 and
          a["long_answer"]["end_token"] >= 0)


def get_first_annotation(e):
  """Returns the first short or long answer in the example.
  Args:
    e: (dict) annotated example.
  Returns:
    annotation: (dict) selected annotation
    annotated_idx: (int) index of the first annotated candidate.
    annotated_sa: (tuple) char offset of the start and end token
        of the short answer. The end token is exclusive.
  """
  positive_annotations = sorted(
      [a for a in e["annotations"] if has_long_answer(a)],
      key=lambda a: a["long_answer"]["candidate_index"])

  for a in positive_annotations:
    if a["short_answers"]:
      idx = a["long_answer"]["candidate_index"]
      start_token = a["short_answers"][0]["start_token"]
      end_token = a["short_answers"][-1]["end_token"]
      return a, idx, (token_to_char_offset(e, idx, start_token),
                      token_to_char_offset(e, idx, end_token) - 1)

  for a in positive_annotations:
    idx = a["long_answer"]["candidate_index"]
    return a, idx, (-1, -1)

  return None, -1, (-1, -1)


def create_example_from_jsonl(line):
    """Creates an NQ example from a given line of JSON."""
    e = json.loads(line, object_pairs_hook=collections.OrderedDict)
    add_candidate_types_and_positions(e)
    annotation, annotated_idx, annotated_sa = get_first_annotation(e)

    question = {"input_text": e["question_text"]}
    answer = {
      "candidate_id": annotated_idx,
      "span_text": "",
      "span_start": -1,
      "span_end": -1,
      "input_text": "long",
    }
    print("annotation: ", annotation)
    print("answer: ", answer)
    print(question)
    print(e.keys())

    context_idxs = [-1]
    context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
    context_list[-1]["text_map"], context_list[-1]["text"] = (
      get_candidate_text(e, -1))
    for idx, _ in candidates_iter(e):
        context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
        context["text_map"], context["text"] = get_candidate_text(e, idx)
        context_idxs.append(idx)
        context_list.append(context)

    if annotated_sa != (-1, -1):
        answer["input_text"] = "short"
        span_text = get_candidate_text(e, annotated_idx)[1]
        answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
        answer["span_start"] = annotated_sa[0]
        answer["span_end"] = annotated_sa[1]
        expected_answer_text = get_text_span(
            e, {
                "start_token": annotation["short_answers"][0]["start_token"],
                "end_token": annotation["short_answers"][-1]["end_token"],
            })[1]
        assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                         answer["span_text"])
    # Add a long answer if one was found.
    elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
        answer["span_text"] = get_candidate_text(e, annotated_idx)[1]
        answer["span_start"] = 0
        answer["span_end"] = len(answer["span_text"])

    single_map = []
    single_context = []
    offset = 0
    for context in context_list:
        single_map.extend([-1, -1])
        single_context.append("[ContextId=%d] %s" %
                          (context["id"], context["type"]))
        offset += len(single_context[-1]) + 1
        if context["id"] == annotated_idx:
          answer["span_start"] += offset
          answer["span_end"] += offset

        # Many contexts are empty once the HTML tags have been stripped, so we
        # want to skip those.
        if context["text"]:
            single_map.extend(context["text_map"])
            single_context.append(context["text"])
            offset += len(single_context[-1]) + 1

    # Assemble example.
    example = {
      "name": e["document_title"],
      "id": str(e["example_id"]),
      "questions": [question],
      "answers": [answer],
      "has_correct_context": annotated_idx in context_idxs
    
    } 
    example["contexts"] = " ".join(single_context)
    example["contexts_map"] = single_map
    if annotated_idx in context_idxs:
        expected = example["contexts"][answer["span_start"]:answer["span_end"]]

    assert expected == answer["span_text"], (expected, answer["span_text"])

    return example
    

def get_nq_examples(gzip_nq):
    gzip_files = [f for f in listdir(gzip_nq)]
    input_data = []
    for gzip_f in gzip_files:
        with gzip.GzipFile(gzip_f) as input_file:
            for line in input_file:
                input_data.append(create_example_from_jsonl(line))
        break

    return input_data

if __name__ == "__main__":
    gzip_nq="/scratch/pmh330/jiant-data/natural_questions/v1.0/train/nq-train-00.jsonl.gz"
    examples = get_examples(gzip_nq)
    print(len(examples))
