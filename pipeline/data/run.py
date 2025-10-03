import argparse
import logging

from pipeline.data.utils import load_artefact
from pipeline.data.enums import Language, DataSplit, DataOperation
from pipeline.constants import ARTEFACTS_DIR
from pipeline.data.train_data import make_hausa_df, make_igbo_df, make_yoruba_df, make_train_dataset
from pipeline.data.eval_test_data import make_eval_test_dataset
from pipeline.data.postprocess import get_audits, postprocess_dataset, sample_data_by_text_length, sample_data_by_language
from pipeline.data.translate import make_english_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_TEST_ROW_COUNT = 2000

def main(args):
    language = args.language.lower()
    language = Language(language)
    if args.operation == "create":
        if language == Language.english:
            raise ValueError("English dataset is not supported for the create operation. Use 'translate' operation instead.")
        
        processors = {
            Language.yoruba: make_yoruba_df,
            Language.igbo: make_igbo_df,
            Language.hausa: make_hausa_df,
        }

        # Not doing the try and catch here because the data used here are primitive datasets that should always exist.
        if args.split == DataSplit.train:
            make_train_dataset(processors.get(language)(), filename=f"{ARTEFACTS_DIR}/{language}_train_dataset.jsonl")
            logger.info(f"Train dataset created for {language} and saved to {ARTEFACTS_DIR}/{language}_train_dataset.jsonl")
        elif args.split == DataSplit.eval:
            make_eval_test_dataset(language, eval_filename=f"{ARTEFACTS_DIR}/{language}_eval_dataset.jsonl", test_filename=None)
            logger.info(f"Eval dataset created for {language} and saved to {ARTEFACTS_DIR}/{language}_eval_dataset.jsonl")
        elif args.split == DataSplit.test:
            make_eval_test_dataset(language, eval_filename=None, test_filename=f"{ARTEFACTS_DIR}/{language}_test_dataset.jsonl")
            logger.info(f"Test dataset created for {language} and saved to {ARTEFACTS_DIR}/{language}_test_dataset.jsonl")
        else:
            raise ValueError(f"Invalid split. Choose from {DataSplit}.")
    elif args.operation == "postprocess":
        if language == Language.english:
            raise ValueError("English dataset is not supported for the postprocess operation. Use 'translate' operation instead.")
        
        try:
            rows = load_artefact(f"{args.language}_{args.split}_dataset.jsonl")
        except ValueError as e:
            msg = f"Could not load dataset for {args.language} and split {args.split}. "
            msg += "Ensure the dataset exists before postprocessing, perhaps you need to run the create operation first."
            raise RuntimeError(msg) from e
        
        try:
            model_name = "gemma3_27b"
            audits = get_audits(f"{args.language}_{model_name}_{args.split}_results.jsonl", n=None)
        except ValueError as e:
            msg = f"Could not load audit results for {args.language}, model {model_name} and split {args.split}. "
            msg += "Ensure the audit results exist before postprocessing, perhaps you need to run the pipeline.llms.ollama_audit pipeline first."
            raise RuntimeError(msg) from e
        
        filtered_lines = postprocess_dataset(rows, audits, args.language)
        filepath = f"{ARTEFACTS_DIR}/filtered_{args.language}_{args.split}_dataset.jsonl"
        with jsonlines.open(filepath, "w") as f_:
            if DataSplit.test in filepath or DataSplit.eval in filepath:
                f_.write_all(sample_data_by_text_length(filtered_lines, n=EVAL_TEST_ROW_COUNT))
            else:
                f_.write_all(filtered_lines)
        logger.info(f"Filtered dataset created for {args.language} and saved to {filepath}.")
    elif args.operation == "translate":
        if language != Language.english:
            raise ValueError("Translate operation is only supported for English dataset.")
        
        try:
            rows = make_english_dataset(args.split)
        except ValueError as e:
            msg = f"Could not create English dataset for split {args.split}. "
            msg += "Ensure the translated datasets exist before creating the English dataset, perhaps you need to run the pipeline.llms.ollama_translation pipeline first."
            raise RuntimeError(msg) from e
        
        filepath = f"{ARTEFACTS_DIR}/filtered_{args.language}_{args.split}_dataset.jsonl"
        with jsonlines.open(filepath, "w") as f_:
            if DataSplit.test in filepath or DataSplit.eval in filepath:
                # Using language aware sampling because the English dataset is a combination of all languages.
                f_.write_all(sample_data_by_language(rows, n=EVAL_TEST_ROW_COUNT))
            else:
                f_.write_all(rows)
        logger.info(f"Filtered dataset translated for {args.language} and saved to {filepath}.")
    else:
        raise ValueError("Invalid operation. Choose from 'creation' or 'postprocess'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=Language, type=str.lower, required=True, help="Language to use for the dataset.")
    parser.add_argument("--split", choices=DataSplit, type=str.lower, default=DataSplit.train, help="Dataset split to postprocess.")
    parser.add_argument("--operation", choices=DataOperation, type=str.lower, default=DataOperation.create, help="Data operation to perform.")
    args = parser.parse_args()

    main(args)
