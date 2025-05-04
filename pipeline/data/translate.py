from copy import deepcopy

import numpy as np

from pipeline.constants import SEED
from pipeline.data.utils import load_artefact
from pipeline.data.utils import fix_negatives
from pipeline.data.enums import Language, DataSplit


def make_english_dataset(split):
    if split not in DataSplit:
        raise ValueError(f"Split must be one of {DataSplit}.")
    
    all_translated_data = []
    languages = [Language.hausa, Language.igbo, Language.yoruba]
    for language in languages:
        rows = load_artefact(f"filtered_{language}_{split}_dataset.jsonl")
        translated_rows = load_artefact(f"{language}_english_gemma3_27b_{split}_results.jsonl")

        new_translated_data = []
        for row, translated_row in zip(rows, translated_rows):
            # Skipping rows where no translation was done
            if row["query"] == translated_row["translation"]:
                continue
            
            new_data = deepcopy(row)
            new_data.update({
                "root_query_text": new_data["query"],
                "query": translated_row["translation"],
                "root_query_language": language,
            })
            new_data.pop("neg", None)
            new_translated_data.append(new_data)

        rng = np.random.default_rng(SEED)
        new_translated_data = [fix_negatives(row_idx, new_translated_data, rng) for row_idx in range(len(new_translated_data))]
        all_translated_data.extend(new_translated_data)

    return all_translated_data
