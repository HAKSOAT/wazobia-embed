from urllib.parse import urlparse

import pandas as pd
from datasets import arrow_dataset

SEED = 42


def extract_domain_name(url):
    try:
        parsed_url = urlparse(url)
        netloc = str(parsed_url.netloc)
        return netloc.strip("www.")
    except ValueError:
        return None


def prepare_wura(dataset):
    if not isinstance(dataset, arrow_dataset.Dataset):
        raise ValueError(f"The parameter `dataset` only accepts `arrow_dataset.Dataset` objects. Got {type(dataset)} instead.")

    expected_columns = {"headline", "content", "category", "url"}
    missing_columns = expected_columns.difference(set(dataset.features))
    if missing_columns:
        raise ValueError(f"The dataset must contain all of the following features: {expected_columns}. Missing features: {missing_columns}")

    domain_counts = {}
    for row in dataset:
        domain = extract_domain_name(row["url"])
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    invalid_domains = {
        "jw.org" # Has really weird links, for example:  https://www.jw.org/yo/elerii-jehofa/kan-si-wa/venezuela/, https://www.jw.org/yo/elerii-jehofa/kan-si-wa/tonga/, https://www.jw.org/yo/elerii-jehofa/kan-si-wa/taiwan/ all have the title "Kan Si Wa"
    }

    is_headline_valid = lambda value: len((value or " ").split()) > 1
    is_url_valid = lambda value: len((value or " ").strip()) > 5
    is_domain_valid = lambda value: domain_counts[value] > 10 and not value in invalid_domains # If the domain does not appear enough times that is a sign that the site is not committed to publishing in the language. So it is probably a weird url or the English was translated using Google translate e.g. https://downloadfacetime.com/facetime/facetime-for-ipad/
    is_text_valid = lambda value: len((value or " ").strip().split()) > 30

    data = []
    for row in dataset:
        if not (is_headline_valid(row["headline"]) \
                and is_url_valid(row["url"]) \
                and is_domain_valid(extract_domain_name(row["url"]))):
            continue

        data.append({
            "title": row["headline"],
            "url": row["url"].strip("/") + "/", "text": row["content"],
            "category": row["category"],
            "source": "wura"
        })

    wura_df = pd.DataFrame(data)
    return wura_df


def wura_remove_validation_rows(df, wura_ds):
    """Checks for rows in df that exist in wura_ds, using the url, then drops them"""
    wura_val_urls = wura_ds["url"]
    wura_val_urls = {url.strip("/") + "/" for url in wura_val_urls}

    def format_url(row):
        if pd.isna(row.url):
            row.url = ""
            return row
        else:
            row.url = row.url.strip("/") + "/"
            return row

    df = df.apply(lambda row: format_url(row), axis=1)
    df = df[~df.url.isin(wura_val_urls)].reset_index(drop=True)

    return df


def unify_datasources(dfs: list, wura_data):
    for df in dfs:
        df.columns = df.columns.str.lower()
        if "sub_topic" not in df.columns:
            df["sub_topic"] = None

    df = pd.concat(dfs)
    df = align_with_wura(df, wura_data)

    # dropna for title and text columns
    key_columns = ["title", "text"]
    df.dropna(subset=key_columns, inplace=True)
    return df


def align_with_wura(df, wura_data):
    df = wura_remove_validation_rows(df, wura_data["validation"])
    # Combined collected dataset with Wura train dataset
    # wura_df = make_wura_df(wura_data["train"])
    wura_df = prepare_wura(wura_data["train"])

    df_urls = set(df.url)
    seen_rows = wura_df.url.isin(df_urls)
    new_wura_df = wura_df[~seen_rows]
    old_wura_df = wura_df[seen_rows]
    df = pd.concat([df, new_wura_df])
    # Extracting the category data available in Wura, so we don't miss out on that data
    df["category"] = df["url"].map(old_wura_df.set_index("url")["category"])
    return df
