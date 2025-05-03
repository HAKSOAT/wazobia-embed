import argparse

import jsonlines

from pipeline.data.utils import load_jsonl
from pipeline.constants import ARTEFACTS_DIR
from pipeline.data.train_data import make_hausa_df, make_igbo_df, make_yoruba_df, make_train_dataset
from pipeline.data.nontrain_data import make_nontrain_dataset
from postprocess import get_audits, postprocess_dataset, sample_data

def main(args):
    language = args.language.lower()
    if args.operation == "creation":
        processors = {
            "yoruba": make_yoruba_df,
            "igbo": make_igbo_df,
            "hausa": make_hausa_df
        }

        if args.split == "train":
            make_train_dataset(processors.get(language)(), filename=f"{ARTEFACTS_DIR}/{language}_train_dataset.jsonl")
        elif args.split == "eval":
            make_nontrain_dataset(language, eval_filename=f"{ARTEFACTS_DIR}/{language}_eval_dataset.jsonl", test_filename=None)
        elif args.split == "test":
            make_nontrain_dataset(language, eval_filename=None, test_filename=f"{ARTEFACTS_DIR}/{language}_test_dataset.jsonl")
        else:
            raise ValueError("Invalid split. Choose from 'train', 'eval', or 'test'.")
    elif args.operation == "postprocess":
        lines = load_jsonl(f"{args.lang}_{args.split}_dataset.jsonl")
        model_name = "gemma3_27b"
        audits = get_audits(f"{args.lang}_{model_name}_{args.split}_results.jsonl", n=None)
        filtered_lines = postprocess_dataset(lines, audits, args.lang)

        fname = f"filtered_{args.lang}_{args.split}_dataset.jsonl"
        with jsonlines.open(fname, "w") as f_:
            if "test" in fname or "eval" in fname:
                f_.write_all(sample_data(filtered_lines, n=2000))
            else:
                f_.write_all(filtered_lines)
            print(f"Written to {fname}")
    else:
        raise ValueError("Invalid operation. Choose from 'creation' or 'postprocess'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["yoruba", "igbo", "hausa"], type=str.lower, required=True, help="Language to use for the dataset.")
    parser.add_argument("--split", choices=["train", "eval", "test"], default="train", help="Dataset split to postprocess.")
    parser.add_argument("--operation", choices=["creation", "postprocess"], default="creation", help="Data operation to perform.")
    args = parser.parse_args()

    main(args)