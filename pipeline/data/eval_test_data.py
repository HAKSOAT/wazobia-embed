from datasets import load_dataset
from sklearn.model_selection import train_test_split

from pipeline.data.wura import prepare_wura
from pipeline.constants import SEED
from pipeline.data.enums import Language
from pipeline.data.constants import WURA_LANG_ID_MAP


def make_eval_test_dataset(language, eval_filename="eval_dataset.jsonl", test_filename="test_dataset.jsonl"):
    if language.lower() not in Language:
        raise ValueError(f"Language must be one of {language}.")
    wura_lang = WURA_LANG_ID_MAP.get(language.lower())
    if not wura_lang:
        raise ValueError(f"Language {language} not found in WURA_LANG_ID_MAP.")
    
    dataset = load_dataset("castorini/wura", wura_lang, level="document", trust_remote_code=True)
    validation_data = dataset.get("validation")

    if not validation_data:
        raise ValueError(f"Dataset {wura_lang} does not have a validation split. Only found {dataset.keys()} splits.")
    
    lang_df = prepare_wura(validation_data)
    lang_df.rename(columns={"text": "pos", "title": "query"}, inplace=True)
    eval_df, test_df = train_test_split(lang_df, test_size=0.4, random_state=SEED, shuffle=True)
    split_dfs = {}
    if eval_filename:
        eval_df.to_json(eval_filename, orient="records", lines=True)
        split_dfs["eval"] = eval_df

    if test_filename:
        test_df.to_json(test_filename, orient="records", lines=True)
        split_dfs["test"] = test_df
    
    return split_dfs
