from urllib.parse import urlparse

import jsonlines
import pandas as pd


def extract_domain_name(url):
    try:
        parsed_url = urlparse(url)
        netloc = str(parsed_url.netloc)
        return netloc.strip("www.")
    except ValueError:
        return None
    

def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as f:
        for line in f:
            data.append(line)
    return data
