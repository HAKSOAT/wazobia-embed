import argparse

from datasets import make_hausa_df, make_igbo_df, make_yoruba_df, make_dataset


def main(args):
    language = args.language.lower()
    processors = {
        "yoruba": make_yoruba_df,
        "igbo": make_igbo_df,
        "hausa": make_hausa_df
    }

    make_dataset(processors.get(language)(), filename=f"{language}_train_dataset.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["Yoruba", "Igbo", "Hausa"], required=True, help="Language to use for the dataset")
    args = parser.parse_args()

    main(args)