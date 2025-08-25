from copy import deepcopy
from functools import reduce
import math
import re
import warnings

import numpy as np

from pipeline.data.enums import DataSource, Language
from pipeline.constants import SEED
from pipeline.text_utils import skip_doc_body
from pipeline.data.utils import extract_domain_name, fix_negatives, load_artefact


def fix_wiki_pos(row):
    query = row["query"].strip()

    if not row.get("root_text"):
        row["root_text"] = deepcopy(row["pos"])

    pos_is_list = isinstance(row["pos"], list)
    pos = row["pos"][0] if pos_is_list else row["pos"]
    substring = f"{query}\n\n"
    pos = pos[len(substring):] if pos.startswith(substring) else pos
    query_split = [re.escape(q) for q in query.split()]
    re_pattern = reduce(lambda a, b:
                        a +
                        r"\s*" + # Mandatory space after each word
                        r"([^\s]+\s*){,2}" + # Potential for 0 to 2 words appearing between query words.
                        b, query_split)
    re_sentence = r"[^\.]*" + re_pattern + r"[^\.]*." # Matches the sentence where the pattern appears
    try:
        re_obj = re.compile(re_sentence, re.IGNORECASE)
    except Exception as exc:
        print("Error compiling the regex pattern: ", re_sentence)
        raise exc
    pos = re_obj.sub("", pos).strip()

    pos = "" if "plánẹ́tì kékeré" in pos else pos # Weird text appearing in Yoruba wiki
    row["pos"] = [pos] if pos_is_list else pos
    return row


def get_audits(filename, n=100, categories=["X", "NLC"]):
    lines = []
    for line in load_artefact(filename):
        if not line:
            continue

        if line["category"] == "PARSE_ERROR":
            re_cat = "|".join(categories)
            match = re.search(
                r"(?:category>(" + re_cat + r")</category)|(?:Category[^a-zA-Z]*(" + re_cat + "))",
                line["message"], re.IGNORECASE
            )
            if match:
                category = [i for i in match.groups() if i][0]
                line["category"] = category

        lines.append(line)
        if n and len(lines) == n:
            break
    return lines

def postprocess_dataset(rows, audits, language):
    language = language.lower()
    if language not in Language:
        raise ValueError(f"Language must be one of {Language}")
    
    language = Language(language)

    def process_row(row):
        domain_name = extract_domain_name(row["url"])
        row = fix_wiki_pos(row) if domain_name.endswith("wikipedia.org") else row
        row["domain"] = domain_name
        return row

    if language != Language.hausa:
        assert len(rows) == len(audits), f"{len(rows)} != {len(audits)}"

    results = []
    for i in range(len(audits)):
        row = rows[i]
        row = process_row(row)
        text = row["pos"][0] if isinstance(row["pos"], list) else row["pos"]

        if not text or skip_doc_body(text, row["query"]):
            continue

        if audits[i]["category"] not in ["NLC", "SKIP", "EMPTYTEXT"]:
            results.append(row)

    # Hausa does not have all the audits done.
    if language == Language.hausa:
        for row in rows[len(audits):]:
            if row["source"] != DataSource.mato:
                continue

            row = process_row(row)
            text = row["pos"][0] if isinstance(row["pos"], list) else row["pos"]

            if not text or skip_doc_body(text, row["query"]):
                continue

            results.append(row)
  
    if "neg" in rows[0]:
        rng = np.random.default_rng(SEED)
        results = [fix_negatives(row_idx, results, rng) for row_idx in range(len(results))]
            
    return results


def sample_data(rows, n=2000):
    if n > len(rows):
        warning_msg = f"All rows were returned at `sample_data` function call. Total number of rows are {len(rows)}, requested to sample {n}"
        warnings.warn(warning_msg)
        return rows

    # Creating a bin based on text length, so we sample from them equally.
    idx_bins = [set(), set(), set(), set()]
    for idx, line in enumerate(rows):
        text = line["pos"][0] if isinstance(line["pos"], list) else line["pos"]
        token_size = len(text.split())
        if token_size <= 512:
            idx_bins[0].add(idx)
        elif token_size <= 1024:
            idx_bins[1].add(idx)
        elif token_size <= 2048:
            idx_bins[2].add(idx)
        else:
            idx_bins[3].add(idx)

    sample_idxs = set()
    attempt = 0
    rng = np.random.default_rng(SEED)
    while len(sample_idxs) < n:
        # In the case where some bins are exhausted earlier than others
        # the desired min is updated based on the remaining number of bins.
        desired_min = math.ceil((n - len(sample_idxs)) / len(idx_bins))
        idx = attempt % len(idx_bins)
        if len(idx_bins[idx]) > desired_min:
            # Converting to a tuple and then to a set is inefficient.
            choices = set(rng.choice(tuple(idx_bins[idx]), desired_min, replace=False))
            sample_idxs.update(choices)
            idx_bins[idx] = idx_bins[idx] - choices
        else:
            sample_idxs.update(idx_bins[idx])
            idx_bins.pop(idx)
        attempt += 1
    samples = [rows[i] for i in sample_idxs]
    return samples[:n]
