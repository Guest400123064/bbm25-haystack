[![test](https://github.com/Guest400123064/bbm25-haystack/actions/workflows/test.yml/badge.svg)](https://github.com/Guest400123064/bbm25-haystack/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Guest400123064/bbm25-haystack/graph/badge.svg?token=IGRIRBHZ3U)](https://codecov.io/gh/Guest400123064/bbm25-haystack)
[![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)
[![Python 3.9](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue.svg)](https://www.python.org/downloads/release/python-390/)

# Better BM25 In-Memory Document Store

An in-memory document store is a great starting point for prototyping and debugging before migrating to production-grade stores like Elasticsearch. However, [the original implementation](https://github.com/deepset-ai/haystack/blob/0dbb98c0a017b499560521aa93186d0640aab659/haystack/document_stores/in_memory/document_store.py#L148) of BM25 retrieval recreates an inverse index for the entire document store __on every new search__. Furthermore, the tokenization method is primitive, only permitting splitters based on regular expressions, making localization and domain adaptation challenging. Therefore, this implementation is a slight upgrade to the default BM25 in-memory document store by implementing incremental index update and incorporation of [SentencePiece](https://github.com/google/sentencepiece) statistical sub-word tokenization.

## Installation

```bash
$ pip install bbm25-haystack
```

Alternatively, you can clone the repository and build from source to be able to reflect changes to the source code:

```bash
$ git clone https://github.com/Guest400123064/bbm25-haystack.git
$ cd bbm25-haystack
$ pip install -e .
```

## Usage

### Quick Start

Below is an example of how you can build a minimal search engine with the `bbm25_haystack` components on their own. They are also compatible with [Haystack pipelines](https://docs.haystack.deepset.ai/docs/creating-pipelines).

```python
from haystack import Document
from bbm25_haystack import BetterBM25DocumentStore, BetterBM25Retriever


document_store = BetterBM25DocumentStore()
document_store.write_documents([
   Document(content="There are over 7,000 languages spoken around the world today."),
   Document(content="Elephants have been observed to behave in a way that indicates a high level of self-awareness, such as recognizing themselves in mirrors."),
   Document(content="In certain parts of the world, like the Maldives, Puerto Rico, and San Diego, you can witness the phenomenon of bio-luminescent waves.")
])

retriever = BetterBM25Retriever(document_store)
retriever.run(query="How many languages are spoken around the world today?")
```

### API References

The initializer takes [three BM25+ hyperparameters](https://en.wikipedia.org/wiki/Okapi_BM25), namely `k1`, `b`, and `delta`, and path to a trained SentencePiece tokenizer `.model` file. All parameters are optional. The default tokenizer is directly copied from [LLaMA-2-7B-32K tokenizer](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/main/tokenizer.model) with a vocab size of 32,000.

## Filtering Logic

The current document store uses `document_matches_filter` shipped with Haystack to perform filtering by default, which is the same as `InMemoryDocumentStore` except that it is DOES NOT support legacy operator names.

However, there is also an alternative filtering logic shipped with this implementation that is more conservative (and unstable at this point). To use this alternative logic, initialize the document store with `haystack_filter_logic=False` Please find comments and implementation details in [`filters.py`](./src/bbm25_haystack/filters.py). TL;DR:

- Comparison with `None`, i.e., missing values, involved will always return `False`, no matter the document attribute value or filter value.
- Comparison with `DataFrame` is always prohibited to reduce surprises.
- No implicit `datetime` conversion from string values.

These differences lead to a few caveats. Firstly, some test cases are overridden to take into account the different expectations. However, this means that passed, non-overridden tests may not be faithful, i.e., the filters behave in the same way as the old implementation while different behaviors are expected. Further, the negation logic needs to be considered again because `False` can now issue from both input check and the actual comparisons. But I think having input processing and comparisons separated makes the filtering behavior more transparent.

## Search Quality Evaluation

This repo has [a simple script](./scripts/benchmark_beir.py) to help evaluate the search quality over [BEIR](https://github.com/beir-cellar/beir/tree/main) benchmark. You need to clone the repository (you can also manually download the script and place it under a folder named `scripts`) and you have to install additional dependencies to run the script.

```bash
$ pip install beir
```

To run the script, you may want to specify the dataset name and BM25 hyperparameters. For example:

```bash
$ python scripts/benchmark_beir.py --datasets scifact arguana --bm25-k1 1.2 --output eval.csv
```

It automatically downloads the benchmarking dataset to `benchmarks/beir`, where `benchmarks` is at the same level as `scripts`. You may also check the help page for more information.

```bash
$ python scripts/benchmark_beir.py --help
```

New benchmarking scripts are expected to be added in the future.

## License

`bbm25-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
