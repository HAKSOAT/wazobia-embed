import tiktoken

def count_tokens(text):
    encoding = tiktoken.get_encoding("gpt2")
    token_ids = encoding.encode(text)
    return len(token_ids)


def truncate_text(text, max_tokens):
    encoding = tiktoken.get_encoding("gpt2")
    token_ids = encoding.encode(text)
    truncated_token_ids = token_ids[:max_tokens]
    truncated_text = encoding.decode(truncated_token_ids)
    return truncated_text

def skip_doc_body(text, query=None):
    # "plánẹ́tì kékeré" is some weird text that appears in Wura's Yoruba wiki
    if ("plánẹ́tì kékeré" in text) or (not text.strip()):
        return True
    if query and text.startswith(query):
        mod_text = text.strip(query).strip()
        if len(mod_text.split()) < 30:
            return True
    elif len(text.split()) < 5:
        return True
    return False