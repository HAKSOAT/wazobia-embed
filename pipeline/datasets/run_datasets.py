import argparse

from datasets import make_hausa_df, make_igbo_df, make_yoruba_df, make_train_dataset, make_nontrain_dataset


def main(args):
    language = args.language.lower()
    processors = {
        "yoruba": make_yoruba_df,
        "igbo": make_igbo_df,
        "hausa": make_hausa_df
    }

    if args.split == "train":
        make_train_dataset(processors.get(language)(), filename=f"{language}_train_dataset.jsonl")
    elif args.split == "eval":
        make_nontrain_dataset(language, eval_filename=f"{language}_eval_dataset.jsonl", test_filename=None)
    elif args.split == "test":
        make_nontrain_dataset(language, eval_filename=None, test_filename=f"{language}_test_dataset.jsonl")
    else:
        raise ValueError("Invalid split. Choose from 'train', 'eval', or 'test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["Yoruba", "Igbo", "Hausa"], required=True, help="Language to use for the dataset.")
    parser.add_argument("--split", choices=["train", "eval", "test"], default="train", help="Dataset split to create.")
    args = parser.parse_args()

    main(args)