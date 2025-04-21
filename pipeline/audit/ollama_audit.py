import argparse
import asyncio
import re
import shutil
import time
import traceback
from pathlib import Path

import jsonlines
import ollama
import tiktoken
from ollama import chat, AsyncClient


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


def skip(text, query=None):
    """Function based on analysis of done audits, to avoid spending time on unnecessary rows"""
    if ("ìgbàjá ástẹ́rọ́ìdì" in text) or (not text.strip()):
        return True
    if query and text.startswith(query):
        mod_text = text.strip(query).strip()
        if len(mod_text.split()) < 30:
            return True
    return False


PROMPT_TOKEN_SIZE = count_tokens(syntactic)


async def chat(language, text, token_limit=2048, model="gemma3:27b"):
  full_prompt = syntactic.format(language=language, text=text)
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
  label = re.search(r"<category>(.*?)</category>", response.message.content)
  label = label.group(1) if label else "PARSE_ERROR"
  reason = re.search(r"<reason>(.*?)</reason>", response.message.content)
  reason = reason.group(1) if reason else "PARSE_ERROR"
  return {"category": label, "reason": reason, "message": response.message.content}


async def main(args):
    language = args.language
    if language.lower() not in ["yoruba", "igbo", "hausa"]:
        raise ValueError("Language must be Yoruba, Igbo or Hausa")

    if args.split == "train":
        dir_ = Path("/content/drive/MyDrive/Side Projects/NaijEmbeddings/datasets/combine_wura_all_langs")
        out_file = f"{language.lower()}_{args.model.replace(':', '_')}_results.jsonl"
        in_file = f"{language.lower()}_train_dataset.jsonl"
    elif args.split == "eval":
        dir_ = Path(f"/content/drive/MyDrive/Side Projects/NaijEmbeddings/datasets/static_wura/{language.lower()}")
        out_file = f"{language.lower()}_{args.model.replace(':', '_')}_eval_results.jsonl"
        in_file = f"{language.lower()}_eval_dataset.jsonl"
    elif args.split == "test":
        dir_ = Path(f"/content/drive/MyDrive/Side Projects/NaijEmbeddings/datasets/static_wura/{language.lower()}")
        out_file = f"{language.lower()}_{args.model.replace(':', '_')}_test_results.jsonl"
        in_file = f"{language.lower()}_test_dataset.jsonl"
    else:
        raise ValueError("Split must be train, eval or test")
    
    if not (dir_/in_file).exists():
        raise RuntimeError(f"Input file {dir_/in_file} does not exist")
                
    shutil.copyfile(dir_ / in_file, in_file)

    # Restoring the file from the output dir, so we continue processing from last stopped.
    if not Path(out_file).exists() and args.restore and Path(dir_ / out_file).exists():
        shutil.copyfile(dir_ / out_file, out_file)

    if Path(out_file).exists():
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
                            if isinstance(line["pos"], list):
                                chat_text = line["pos"][0]
                            elif isinstance(line["pos"], str):
                                chat_text = line["pos"]
                            else:
                                raise ValueError("pos must be a string or a list of strings")

                            if not chat_text:
                                results[idx] = {"category": "EMPTYTEXT", "reason": "EMPTYTEXT", "message": "EMPTYTEXT"}
                            elif skip(chat_text, line["query"]):
                                results[idx] = {"category": "SKIP", "reason": "SKIP", "message": "SKIP"}
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
                shutil.copyfile(out_file, dir_ / out_file)

                num_results += len(results)
                print(f"Done {num_results}")
                lines = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma3:27b")
    parser.add_argument("--language", type=str, default="Yoruba")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--split", choices=["train", "eval", "test"], default="train")
    parser.add_argument("--restore", type=bool, default=True)
    args = parser.parse_args()

    ollama.pull(args.model)
    asyncio.run(main(args))
