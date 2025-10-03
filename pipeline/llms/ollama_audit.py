import argparse
import asyncio
import re
import time
import traceback
from pathlib import Path

import jsonlines
import ollama

from ollama import AsyncClient

from pipeline.constants import ARTEFACTS_DIR
from pipeline.data.utils import load_artefact
from pipeline.text_utils import count_tokens, truncate_text, skip_doc_body
from pipeline.data.enums import DataSplit, Language


syntactic = """
You are a linguist, vast in English, Yoruba, Igbo and Hausa languages.

You are tasked with looking at a text to tell if there are syntactic challenges with them.

There are the following categories: NLC, X.

NLC: Short for Not Language Content, this means the text is not a useful text content. For example it has random characters, only has emojis, is extremely short of have meaning, has only spaces or numbers, etc.
X: The text is fine and does not fit into any of the categories above.

So, here is your linguist task:

Given that code mixing is allowed, classify the following {language} language text into the NLC or X category (you must stick to those categories), give a less than 20 word reasoning for your choice.
<text>
{text}
</text>

Populate the following tags with your answer:

<category></category>
<reason></reason>
"""

PROMPT_TOKEN_SIZE = count_tokens(syntactic)


async def chat(language, text, token_limit=2048, model="gemma3:27b"):
    full_prompt = syntactic.format(language=language.capitalize(), text=text)
    token_size = count_tokens(full_prompt)
    if token_size > token_limit:
        remaining_tokens = token_limit - PROMPT_TOKEN_SIZE
        gap_size = 500
        final_text = truncate_text(text, remaining_tokens-gap_size)
        final_prompt = syntactic.format(language=language.capitalize(), text=final_text)
    else:
        final_prompt = full_prompt

    message = {'role': 'user', 'content': final_prompt}
    response = await AsyncClient().chat(model=model, messages=[message])
    label = re.search(r"<category>(.*?)</category>", response.message.content)
    label = label.group(1) if label else "PARSE_ERROR"
    reason = re.search(r"<reason>(.*?)</reason>", response.message.content)
    reason = reason.group(1) if reason else "PARSE_ERROR"
    return {"category": label, "reason": reason, "message": response.message.content}


async def main(args):
    language = args.language
    if language.lower() not in DataSplit:
        raise ValueError(f"Language must be one of {Language}. Got {language.lower()}")
                
    if Path(args.output_path).exists():
        with jsonlines.open(args.output_path) as f:
            # allow_none is handling a bug in previous processing where some lines were None
            start_num_rows = sum(1 for line in f.iter(allow_none=True) if line)
    else:
        start_num_rows = 0

    results = None
    retries = 5

    input_rows = load_artefact(args.input_path)
    num_input_rows = len(input_rows)

    batch = []
    for count, line in enumerate(input_rows[start_num_rows:], start=start_num_rows+1):
        batch.append(line)
        if not (len(batch) == args.batch_size or count == num_input_rows):
            continue

        # Setting up retries because the run_ollama.sh is set to restart Ollama after some intervals.
        # Hence, we need to retry the request if it fails on such occassions.
        last_exc = None
        for _ in range(retries):
            try:
                results = [None] * min(args.batch_size, len(batch))
                tasks = []
                for idx, line in enumerate(batch):
                    if isinstance(line["pos"], list):
                        chat_text = line["pos"][0]
                    elif isinstance(line["pos"], str):
                        chat_text = line["pos"]
                    else:
                        raise ValueError("pos must be a string or a list of strings")

                    if not chat_text:
                        results[idx] = {"category": "EMPTYTEXT", "reason": "EMPTYTEXT", "message": "EMPTYTEXT"}
                    elif skip_doc_body(chat_text, line["query"]):
                        results[idx] = {"category": "SKIP", "reason": "SKIP", "message": "SKIP"}
                    else:
                        tasks.append(chat(language, chat_text, model=args.model))

                responses = await asyncio.gather(*tasks)
                for idx, result in enumerate(results):
                    if result is None:
                        results[idx] = responses.pop(0)
                
                with jsonlines.open(args.output_path, 'a') as f:
                    f.write_all(results)

                print(f"Done {count}")
                batch = []
                break
            except Exception as exc:
                print(line)
                print(f"Error: {traceback.format_exc()}")
                last_exc = exc
                time.sleep(10)
        else:
            raise Exception("All retries failed") from last_exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma3:27b")
    parser.add_argument("--language", choices=Language, default="yoruba", type=str.lower)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--split", choices=DataSplit, default="train", type=str.lower)
    args = parser.parse_args()

    language = args.language.lower()
    if args.split in DataSplit:
        args.input_path = f"{ARTEFACTS_DIR}/{language.lower()}_{args.split}_dataset.jsonl"
        args.output_path = f"{ARTEFACTS_DIR}/{language.lower()}_{args.model.replace(':', '_')}_{args.split}_results.jsonl"
    else:
        raise ValueError(f"Split must be one of {DataSplit}. Got {args.split}")

    ollama.pull(args.model)
    asyncio.run(main(args))
