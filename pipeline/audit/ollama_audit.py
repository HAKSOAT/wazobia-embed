import argparse
import asyncio
import re
import shutil
import time
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


PROMPT_TOKEN_SIZE = count_tokens(syntactic)


async def chat(language, text, token_limit=2048, model="gemma3:27b"):
  full_prompt = syntactic.format(language=language, text=text)
  token_size = count_tokens(full_prompt)
  if token_size > token_limit:
    remaining_tokens = token_limit - PROMPT_TOKEN_SIZE
    gap_size = 100
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

    out_file = f"{language.lower()}_{args.model.replace(':', '_')}_results.jsonl"
    in_file = f"{language.lower()}_train_dataset.jsonl"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Restoring the file from the output dir, so we continue processing from last stopped.
    if args.restore and Path(output_dir / out_file).exists():
        shutil.copyfile(output_dir / out_file, out_file)

    if Path(out_file).exists():
        with jsonlines.open(out_file) as f:
            num_results = sum(1 for line in f)
    else:
        num_results = 0

    results = []
    lines = []
    count = 0
    retries = 5

    with jsonlines.open(in_file, 'r') as f:
        for line in f:
            count += 1
            if count <= num_results:
                continue

            lines.append(line)
            if len(lines) == args.batch_size:
                error = None
                for _ in range(retries):
                    try:
                        responses = await asyncio.gather(*[chat(language, line["pos"][0], model=args.model) for line in lines])
                        results.extend(responses)
                        break
                    except Exception as exc:
                        print(f"Error: {exc}")
                        error = exc
                        time.sleep(10)
                else:
                    raise Exception("All retries failed") from error

                with jsonlines.open(out_file, 'a') as f:
                    f.write_all(results)
                shutil.copyfile(out_file, output_dir / out_file)

                num_results += len(results)
                print(f"Done {num_results}")
                lines = []
                results = []

    with jsonlines.open(out_file, 'a') as f:
        f.write_all(results)
    shutil.copyfile(out_file, output_dir / out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma3:27b")
    parser.add_argument("--language", type=str, default="Yoruba")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/Side Projects/NaijEmbeddings/datasets/combine_wura_all_langs")
    parser.add_argument("--restore", type=bool, default=True)
    args = parser.parse_args()

    ollama.pull(args.model)
    asyncio.run(main(args))
