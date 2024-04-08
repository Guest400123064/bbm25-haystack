[![test](https://github.com/Guest400123064/bbm25-haystack/actions/workflows/test.yml/badge.svg)](https://github.com/Guest400123064/bbm25-haystack/actions/workflows/test.yml)
[![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)
<!-- [![Coverage Status](https://coveralls.io/repos/github/Guest400123064/bbm25-haystack/badge.svg?branch=main)](https://coveralls.io/github/Guest400123064/bbm25-haystack?branch=main) -->

# Better BM25 In-memory Document Store

In-memory document store is a great starting point for prototyping and debugging, before migrating to production-grade stores like ElasticSeach. However, [the original implementation](https://github.com/deepset-ai/haystack/blob/0dbb98c0a017b499560521aa93186d0640aab659/haystack/document_stores/in_memory/document_store.py#L148) of BM25 retrieval recreates an inverse index for entire document store on every new search. Further, the tokenization method is primitive, only permitting splitters based on regular expressions, making localization and domain adaptation challenging. Therefore, this implementation is a slight upgrade to the default BM25 in-memory document store by implementing incremental index update and incorporation of [SentencePiece](https://github.com/google/sentencepiece) statistical sub-word tokenization.

## Installation

This package has not yet published to PyPI. Please directly install the package from `main` branch using:

```bash
pip install git+https://github.com/Guest400123064/bbm25-haystack.git@main
```

## Usage

Initializer takes [three BM25+ hyperparameters](https://en.wikipedia.org/wiki/Okapi_BM25), namely `k1`, `b`, and `delta`, and one path to a trained SentencePiece tokenizer `.model` file. All parameters are optional. The default tokenizer is directly copied from [this SentencePiece test tokenizer](https://github.com/google/sentencepiece/blob/master/python/test/test_model.model) with a vocab size of 1000.

```python
from haystack import Document
from bbm25_haystack import BetterBM25DocumentStore, BetterBM25Retriever


document_store = BetterBM25DocumentStore()
document_store.write_documents([
   Document(content="There are over 7,000 languages spoken around the world today."),
   Document(content="Elephants have been observed to behave in a way that indicates a high level of self-awareness, such as recognizing themselves in mirrors."),
	Document(content="In certain parts of the world, like the Maldives, Puerto Rico, and San Diego, you can witness the phenomenon of bioluminescent waves.")
])

retriever = BetterBM25Retriever(document_store)
retriever.run(query="How many languages are spoken around the world today?")
```

<!-- This Github repository is a template that can be used to create custom document stores to extend
the new [Haystack](https://github.com/deepset-ai/haystack/) API available from version 2.0.

While the new API is still under active development, the new "Store" architecture is quite stable
and we are encouraging early adopters to contribute their custom document stores.

## Template features

By creating a new repo using this template, you'll get the following advantages:
- Ready-made code layout and scaffold to build a custom document store.
- Support for packaging and distributing the code through Python wheels using Hatch.
- Github workflow to build and upload a package when tagging the repo.
- Github workflow to run the tests on Pull Requests.

## How to use this repo

1. Create a new repository starting from this template. If you never used this feature before, you
   can find more details in [Github docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).
2. If possible, follow the convention `technology-haystack` for the name of the new repository,
   where `technology` can be for example the name of a vector database you're integrating.
3. Rename the package `src/example_store` to something more meaningful and adjust the Python
   import statements.
4. Edit `pyproject.toml` and replace any occurrence of `example_store` and `example-store` according
   to the name you chose in the previous steps.
5. Search the whole codebase for the string `#FIXME`, that's where you're supposed to change or add
   code specific for the database you're integrating.
6. If Apache 2.0 is not suitable for your needs, change the software license.

When your custom document store is ready and working, feel free to add it to the list of available
[Haystack Integrations](https://haystack.deepset.ai/integrations) by opening a Pull Request in
[this repo](https://github.com/deepset-ai/haystack-integrations).


## Test

You can use `hatch` to run the linters:

```console
~$ hatch run lint:all
cmd [1] | ruff .
cmd [2] | black --check --diff .
All done! ✨ 🍰 ✨
6 files would be left unchanged.
cmd [3] | mypy --install-types --non-interactive src/example_store tests
Success: no issues found in 6 source files
```

Similar for running the tests:

```console
~$ hatch run cov
cmd [1] | coverage run -m pytest tests
...
```

## Build

To build the package you can use `hatch`:

```console
~$ hatch build
[sdist]
dist/example_store-0.0.1.tar.gz

[wheel]
dist/example_store-0.0.1-py3-none-any.whl
```

## Release

To automatically build and push the package to PyPI, you need to set a repository secret called `PYPI_API_TOKEN`
containing a valid token for your PyPI account.
Then set the desired version number in `src/example_store/__about__.py` and tag the commit using the format
`vX.Y.Z`. After pushing the tag, a Github workflow will start and take care of building and releasing the package.
-->

## License

`bbm25-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
