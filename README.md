# Improving BGE-M3 Multilingual Dense Embeddings for Nigerian Low Resource Languages

This project extends the [BGE-M3](https://github.com/FlagOpen/FlagEmbedding/blob/master/docs/source/bge/bge_m3.rst) on more data for Nigerian native languages: Yoruba, Igbo and Hausa.

## Setting up the Environment

This is a Python project based on the `uv` package manager, so you need to [run its installation script](https://docs.astral.sh/uv/getting-started/installation/) if you do not already have it installed.

The repo can be set up by running:

```
git clone https://github.com/HAKSOAT/wazobia-embed.git
uv sync
```

## Running the Model

The model weights are currently on Huggingface at [abdulmatinomotoso/bge-finetuned](https://huggingface.co/abdulmatinomotoso/bge-finetuned).

You can then use it to generate embeddings via:

```
import torch
from FlagEmbedding import BGEM3FlagModel

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
half_precision = False # or True if you really want
# Load the abdulmatinomotoso/bge-finetuned model
model = BGEM3FlagModel('abdulmatinomotoso/bge-finetuned', use_fp16=half_precision, devices=[device])

# Example text to generate embeddings for
documents = [
    "Eyi jẹ́ gbolohun àpẹẹrẹ",
    "Eyi kì í ṣe gbolohun àpẹẹrẹ",

    "Nke a bụ nkebisiokwu atụ",
    "Nke a abụghị nkebisiokwu atụ",

    "Wannan jimla ce ta misali",
    "Wannan ba jimla ce ta misali ba"
]

query = "Where is the example?"

# Generate embeddings
sparse_embeddings = False # The sparse embeddings are not useful as is, they will require some work to make the align with the entire model.
multivec_embeddings = False # The multivector embeddings are still useful despite training only for dense, but this sample uses only dense embeddings
dense_embeddings = True

doc_embeddings = model.encode(documents, return_sparse=sparse_embeddings, return_dense=dense_embeddings, return_colbert_vecs=multivec_embeddings)["dense_vecs"]

query_embeddings = model.encode([query], return_sparse=sparse_embeddings, return_dense=dense_embeddings, return_colbert_vecs=multivec_embeddings)["dense_vecs"]

similarity_scores = query_embeddings @ doc_embeddings.T
print(similarity_scores)
# array([[
#     0.4360781, # Higher similarity than the opposite text
#     0.40966046, 
    
#     0.4775736, # Higher similarity than the opposite text
#     0.44197953, 
    
#     0.4514127, # Higher similarity than the opposite text
#     0.41557986
# ]], dtype=float32)
```

## The Dataset

The datasets used in this work are currently on Google Drive. You can find the drive IDs for the various files in `pipeline/data/constants.py`.

The final datasets used in training, evaluating and testing the models are of the format: `filtered_<language>_<split>_dataset.jsonl`.

You can then download them using gdown (should be installed if you followed the initial setup) via the command:

```
gdown <drive_id>
```

For example, the `filtered_english_test_dataset.jsonl` dataset can be downloaded using:

```
gdown 1UDAxYEGOXRLMjEp9me3iAiuKKvUXwlwv
```

## Pipelines

Running the pipelines requires the `pipeline` extra on `uv`, you can set this up by running:

```
uv sync --extra pipeline
```

While the datasets already exist and can be downloaded based on [the dataset section](#the-dataset), it is possible that the data in drive might be deleted for storage cost reasons. This would leave only the primitive datasets from which the rest were derived. These pipelines show how to recompute the dataset.

### Running the Data Pipelines

This section contains data pipelines used to create the datasets used when running the experiments. The dataset section contains the actual data, but this section is necessary for reproducibility as the datasets might be deleted in the future.

*NB: For all the commands in this section, replace the anchor bracket variables with appropriate values.*

The data pipelines can generate train, eval and test datasets for the Yoruba, Igbo and Hausa languages.

There are also a couple of operations: `create`, `postprocess` and `translate`.

**create**

This operation creates the initial datasets. These datasets will likely only be useful when working from the original datasets, note that this will contain a lot of unwanted data artifacts.

It can be run with the command:

```
python -m pipeline.data.run --language <language> --split <split> --operation create
```

**postprocess**

This operation creates the final datasets i.e. the ones used in training the published model. These datasets are more likely what you want to work with as they go through processing of the initial dataset.

It can be run with the command:

```
python -m pipeline.data.run --language <language> --split <split> --operation postprocess
```

**translate**

This operation creates the English datasets for the specified languages. English datasets here means documents from the final datasets but with English queries.

It can be run with the command:

```
python -m pipeline.data.run --language <language> --split <split> --operation translate
```

### Running the LLM Pipelines

Large Language Models were used in this project to help with dataset creation.

The language models used in these pipelines were run using [Ollama](https://ollama.com/), hence there is a need to install the software first, then run it via:

```
chmod +x && pipeline/llms/run_ollama.sh
```

*NB: For all the commands in this section, replace the anchor bracket variables with appropriate values.*

The two tasks for which LLMs were used are: dataset auditing and text translation.

**dataset auditing**

This pipeline checks for semantic issues with the dataset e.g. the Igbo dataset containing English rows, weird repetitions, and various other data artifacts that are hard to find by regular analysis.

It can be run with the command:

```
python -m pipeline.llms.ollama_audit --model <model> --language <language> --split <split>
```

You can also find the other parameters for the pipeline using `python -m pipeline.llms.ollama_audit --help`.

**text translation**

This pipeline create a translation dataset that makes it possible to train English query to target language document embeddings. It translates Yoruba, Igbo, Hausa queries into English queries.

It can be run with the command:

```
python -m pipeline.llms.ollama_translation --model <model> --language <language> --split <split>
```

You can also find the other parameters for the pipeline using `python -m pipeline.llms.ollama_translation --help`.

## Citation

If you find this repository useful, please consider giving a star :star: and citation

```
@misc{wazobia_embed,
  title={Improving BGE-M3 Multilingual Dense Embeddings for Nigerian Low Resource Languages},
  author={Abdulmatin Omotoso, Habeeb Shopeju, Adejumobi Joshua, and Shiloh Oni},
  year={2025},
  howpublished={\url{https://github.com/HAKSOAT/wazobia-embed}}
}
```

## License
wazobia-embed is licensed under the [MIT License](https://github.com/HAKSOAT/wazobia-embed/blob/main/LICENSE).