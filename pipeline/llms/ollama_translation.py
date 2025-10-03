import argparse
import asyncio
import re
import shutil
import time
import traceback
from pathlib import Path

import jsonlines
import ollama

from ollama import AsyncClient

from pipeline.constants import ARTEFACTS_DIR
from pipeline.data.enums import DataSplit
from pipeline.text_utils import count_tokens, truncate_text


syntactic = """
You are a linguist, vast in English, Yoruba, Igbo and Hausa languages.

You are tasked with translating into English.

Given some text that is likely in {language}, if not convert it from whatever language it is in to English, if it is in English already, make no changes.

Here is the text:
<text>{text}</text>

Populate the following tags with your answer:
<translation></translation>

Give a one sentence explanation of the translation action taken with the following tag:
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
    final_prompt = syntactic.format(language=language, text=final_text)
  else:
    final_prompt = full_prompt

  message = {'role': 'user', 'content': final_prompt}
  response = await AsyncClient().chat(model=model, messages=[message])
  translation = re.search(r"<translation>(.*?)</translation>", response.message.content)
  translation = translation.group(1) if translation else "PARSE_ERROR"
  reason = re.search(r"<reason>(.*?)</reason>", response.message.content)
  reason = reason.group(1) if reason else "PARSE_ERROR"
  return {"translation": translation, "reason": reason, "message": response.message.content}


async def main(args):
    language = args.language
    if language.lower() not in ["yoruba", "igbo", "hausa"]:
        raise ValueError("Language must be Yoruba, Igbo or Hausa")

    in_file = Path(args.input_path)
    out_file = Path(args.output_path)

    if not in_file.exists():
        raise RuntimeError(f"Input file {in_file} does not exist.")

    if out_file.exists():
        with jsonlines.open(out_file) as f:
            # allow_none is handling a bug in previous processing where
            # some lines were None
            num_results = sum(1 for line in f.iter(allow_none=True) if line)
    else:
        num_results = 0

    results = None
    lines = []
    count = 0
    retries = 5

    with jsonlines.open(in_file, 'r') as f:
        # allow_none is handling a bug in previous processing where
        # some lines were None
        num_rows = sum(1 for line in f.iter(allow_none=True) if line)

    with jsonlines.open(in_file, 'r') as f:
        for line in f.iter(allow_none=True):
            count += 1
            if count <= num_results:
                continue

            lines.append(line)
            if len(lines) == args.batch_size or count == num_rows:
                results = [None] * min(args.batch_size, len(lines))
                error = None
                for _ in range(retries):
                    try:
                        tasks = []
                        for idx, line in enumerate(lines):
                            chat_text = line["query"]

                            if not chat_text:
                                results[idx] = {"category": "EMPTYTEXT", "reason": "EMPTYTEXT", "message": "EMPTYTEXT"}
                            else:
                                tasks.append(chat(language, chat_text, model=args.model))

                        responses = await asyncio.gather(*tasks)
                        for idx, result in enumerate(results):
                            if result is None:
                                results[idx] = responses.pop(0)
                        break
                    except Exception as exc:
                        results = [None] * min(args.batch_size, len(lines))
                        print(line)
                        print(f"Error: {traceback.format_exc()}")
                        error = exc
                        time.sleep(10)
                else:
                    raise Exception("All retries failed") from error

                with jsonlines.open(out_file, 'a') as f:
                    f.write_all(results)

                num_results += len(results)
                print(f"Done {num_results}")
                lines = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma3:27b")
    parser.add_argument("--language", type=str, default="Yoruba")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--split", choices=["train", "eval", "test"], default="train")
    args = parser.parse_args()

    language = args.language.lower()
    if args.split in DataSplit:
        args.input_path = f"{ARTEFACTS_DIR}/filtered_{language.lower()}_{args.split}_dataset.jsonl"
        args.output_path = f"{ARTEFACTS_DIR}/{language.lower()}_english_{args.model.replace(':', '_')}_{args.split}_results.jsonl"
    else:
        raise ValueError(f"Split must be one of {DataSplit}. Got {args.split}")

    ollama.pull(args.model)
    asyncio.run(main(args))
