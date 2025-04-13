import pandas as pd
import numpy as np
from datasets import load_dataset

from pipeline.datasets.utils import unify_datasources


def make_dataset(df, duplicate_rows=False, filename="train_dataset.jsonl"):
    """In this version of make dataset, no longer split into train and eval, because eval and test datasets are currently gotten from wura."""
    df_count = len(df)
    df["neg"] = None
    def pick_negative_values(row):
        picked = False
        neg = row.neg
        if not neg:
            size = 7
            neg = []
        else:
            neg = [neg]
            size = 6

        while not picked:
            indexes = np.random.choice(df_count, size=size, replace=False)
            if row.name not in indexes:
                picked = True

        new_neg = neg + df.iloc[indexes].pos.tolist()
        return new_neg

    df.rename(columns={"text": "pos", "title": "query"}, inplace=True)
    df["neg"] = df.apply(lambda row: pick_negative_values(row), axis=1)
    # Extracting subtopics and using them as a query in duplicate rows
    rows_wo_subtopic = df["sub_topic"].isna()
    if duplicate_rows:
        sub_topic_df = df[~rows_wo_subtopic].copy()
        sub_topic_df.loc[:, "query"] = sub_topic_df.loc[:, "sub_topic"]
        df = pd.concat([df, sub_topic_df])
    else:
        df.loc[~rows_wo_subtopic, "query"] = df[~rows_wo_subtopic].sub_topic

    # The BGE M3 expects a list of values
    df["pos"] = df["pos"].apply(lambda x: [x])
    df = df.loc[:, ["query", "pos", "neg"]]
    seed = 42
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_json(filename, orient="records", lines=True)


def make_hausa_df():
    wura_data = load_dataset("castorini/wura", "hau", level="document", verification_mode="no_checks", trust_remote_code=True)
    df1 = pd.read_csv("hausa_mato_81k.tsv", delimiter="\t")
    # Key to note that drop duplicates is being done.
    # Later on, this should be handled better. DUplicates are being dropped here to avoid potentially
    # using the same link as a negative, as at the moment, negatives are being sampled using n-1.
    df1 = df1.drop_duplicates(["link"])
    df1.rename(columns={"link": "url"}, inplace=True)
    df = unify_datasources([df1], wura_data)

    return df


def make_yoruba_df():
    """Combines collected dataset with the wura dataset, ensuring the urls from collected dataset do not appear in wura validation."""
    wura_data = load_dataset("castorini/wura", "yor", level="document", verification_mode="no_checks", trust_remote_code=True)
    df1 = pd.read_csv('alaroye_mato_10k.tsv', delimiter="\t")
    df2 = pd.read_csv('von_mato_6k.tsv', delimiter="\t")
    df3 = pd.read_csv('masakhanews_1k.tsv', delimiter="\t")

    df2.rename(columns={'link': 'url'}, inplace=True)
    df3.rename(columns={'headline': 'title'}, inplace=True)

    df = unify_datasources([df1, df2, df3], wura_data)
    return df


def make_igbo_df():
    """Combines collected dataset with the wura dataset, ensuring the urls from collected dataset do not appear in wura validation."""
    wura_data = load_dataset("castorini/wura", "ibo", level="document", verification_mode="no_checks", trust_remote_code=True)
    df1 = pd.read_csv("igbo_mato_3k.tsv", delimiter="\t")

    df1.rename(columns={"link": "url"}, inplace=True)
    df = unify_datasources([df1], wura_data)

    return df