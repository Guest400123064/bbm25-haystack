# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
import heapq
import math
import os
from collections import Counter, deque
from collections.abc import Iterable
from itertools import chain
from typing import Any, Final, Optional, Union

import pandas as pd
from haystack import Document, default_from_dict, default_to_dict, logging
from haystack.document_stores.errors import (
    DuplicateDocumentError,
    MissingDocumentError,
)
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.filters import document_matches_filter
from sentencepiece import SentencePieceProcessor  # type: ignore

from bbm25_haystack.filters import apply_filters_to_document

logger = logging.getLogger(__name__)


def _n_grams(seq: Iterable[str], n: int):
    """
    Returns a sliding window (of width n) over data from the iterable.

    :param seq: the input token sequence.
    :type seq: Iterable[str]
    :param n: the window size.
    :type n: int

    :return: the n-gram window generator.
    :rtype: Generator[tuple[str], None, None]
    """
    it = iter(seq)
    wd = deque((next(it, None) for _ in range(n)), maxlen=n)

    yield tuple(wd)
    for el in it:
        wd.append(el)
        yield tuple(wd)


class BetterBM25DocumentStore:
    """
    An in-memory document store intended to improve the default BM25 document
    store shipped with Haystack.
    """

    default_sp_file: Final = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "default.model"
    )

    def __init__(
        self,
        *,
        k: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        sp_file: Optional[str] = None,
        n_grams: Union[int, tuple[int, int]] = 1,
        haystack_filter_logic: bool = True,
    ) -> None:
        """
        Creates a new BetterBM25DocumentStore instance.

        An in-memory document store intended to improve the default
        BM25 document store shipped with Haystack. The default store
        recompute the index for the entire document store for every
        in-coming query, which is significantly inefficient. This
        store aims to improve the efficiency by pre-computing the
        index for all documents in the store and only do incremental
        updates when new documents are added or removed. Further, it
        leverages a SentencePiece model to tokenize the input text
        to allow more flexible and dynamic tokenization adapted to
        domain-specific text.

        :param k: the k1 parameter in BM25+ formula.
        :type k: float, optional
        :param b: the b parameter in BM25+ formula.
        :type b: float, optional
        :param delta: the delta parameter in BM25+ formula.
        :type delta: float, optional
        :param sp_file: the SentencePiece model file to use for
            tokenization.
        :type sp_file: Optional[str], optional
        :param n_grams: the n-gram window size.
        :type n_grams: Optional[Union[int, tuple[int, int]]], optional
        :param haystack_filter_logic: Whether to use the Haystack
            filter logic or the one implemented in this store,
            which is more conservative.
        :type haystack_filter_logic: bool, optional
        """
        self.k = k
        self.b = b

        # Adjust the delta value so that we can bring the `(k1 + 1)`
        # term out of the 'term frequency' term in BM25+ formula and
        # delete it; this will not affect the ranking
        self.delta = delta / (self.k + 1.0)

        self._parse_sp_file(sp_file=sp_file)
        self._parse_n_grams(n_grams=n_grams)

        self._haystack_filter_logic = haystack_filter_logic
        self._filter_func = (
            document_matches_filter
            if self._haystack_filter_logic
            else apply_filters_to_document
        )

        self._avg_doc_len: float = 0.0
        self._freq_doc: Counter = Counter()
        self._index: dict[str, tuple[Document, dict[tuple[str], int], int]] = {}

    def _parse_sp_file(self, sp_file: Optional[str]) -> None:
        self._sp_file = sp_file

        if sp_file is None:
            self._sp_inst = SentencePieceProcessor(model_file=self.default_sp_file)
            return

        if not os.path.exists(sp_file) or not os.path.isfile(sp_file):
            msg = (
                f"Tokenizer model file '{sp_file}' not accessible; "
                f"fallback to default {self.default_sp_file}."
            )
            logger.warn(msg)
            self._sp_inst = SentencePieceProcessor(model_file=self.default_sp_file)
            return

        try:
            self._sp_inst = SentencePieceProcessor(model_file=sp_file)
        except Exception as exc:
            msg = (
                f"Failed to load tokenizer model file '{sp_file}': {exc}; "
                f"fallback to default {self.default_sp_file}."
            )
            logger.error(msg)
            self._sp_inst = SentencePieceProcessor(model_file=self.default_sp_file)

    def _parse_n_grams(self, n_grams: Optional[Union[int, tuple[int, int]]]) -> None:
        self._n_grams = n_grams

        if isinstance(n_grams, int):
            self._n_grams_min = 1
            self._n_grams_max = n_grams
            return

        if isinstance(n_grams, tuple):
            self._n_grams_min, self._n_grams_max = n_grams
            if not all(isinstance(n, int) for n in n_grams):
                msg = f"Invalid n-gram window size: {n_grams}."
                raise ValueError(msg)
            return

        msg = f"Invalid n-gram window size: {n_grams}; expected int or tuple."
        raise ValueError(msg)

    def _tokenize(self, texts: Union[str, list[str]]) -> list[list[tuple[str]]]:
        """
        Tokenize input text using SentencePiece model.

        The input text can either be a single string or a list of strings,
        such as a single user query or a group of raw document. The tokenized
        text will be augmented into set of n-grams based.

        :param texts: the input text to tokenize.
        :type texts: Union[str, list[str]]

        :return: the tokenized text, with n-grams augmented.
        :rtype: list[list[tuple[str]]]
        """

        def _augment_to_n_grams(tokens: list[str]) -> list[tuple[str]]:
            it = (
                _n_grams(tokens, n)
                for n in range(self._n_grams_min, self._n_grams_max + 1)
            )
            return list(chain(*it))

        if isinstance(texts, str):
            texts = [texts]
        return [
            _augment_to_n_grams(tokens)
            for tokens in self._sp_inst.encode(texts, out_type=str)
        ]

    def _compute_bm25plus(
        self,
        query: str,
        documents: list[Document],
    ) -> list[tuple[Document, float]]:
        """
        Calculate the BM25+ score for all documents in this index.

        :param query: the query to calculate the BM25+ score for.
        :type query: str
        :param documents: the pool of documents to calculate the BM25+ score for.
        :type documents: list[Document]

        :return: the BM25+ scores for all documents.
        :rtype: list[tuple[Document, float]]
        """
        cnt = lambda ng: self._freq_doc.get(ng, 0)
        idf = {
            ng: math.log(1 + (len(self._index) - cnt(ng) + 0.5) / (cnt(ng) + 0.5))
            for ng in self._tokenize(query)[0]
        }

        sim = []
        for doc in documents:
            _, freq, doc_len = self._index[doc.id]
            doc_len_scaled = doc_len / self._avg_doc_len

            scr = 0.0
            for token, idf_val in idf.items():
                freq_term = freq.get(token, 0.0)
                freq_damp = self.k * (1 + self.b * (doc_len_scaled - 1))

                tf_val = freq_term / (freq_term + freq_damp) + self.delta
                scr += idf_val * tf_val

            sim.append((doc, scr))

        return sim

    def _retrieval(
        self,
        query: str,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents from the store using the given query.

        :param query: the query to search for.
        :type query: str
        :param filters: the filters to apply to the document list.
        :type filters: Optional[dict[str, Any]]
        :param top_k: the number of documents to return.
        :type top_k: int

        :return: the top-k documents and corresponding sim score.
        :rtype: list[tuple[Document, float]]
        """
        documents = self.filter_documents(filters)
        if not documents:
            return []

        sim = self._compute_bm25plus(query, documents)
        if top_k is None:
            return sorted(sim, key=lambda x: x[1], reverse=True)
        return heapq.nlargest(top_k, sim, key=lambda x: x[1])

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :return: the number of documents in the store.
        :rtype: int
        """
        return len(self._index)

    def filter_documents(
        self, filters: Optional[dict[str, Any]] = None
    ) -> list[Document]:
        """
        Filter documents in the store using the given filters.

        :param filters: the filters to apply to the document list.
        :type filters: Optional[dict[str, Any]]

        :return: the list of documents that match the given filters.
        :rtype: list[Document]
        """
        if filters is None or not filters:
            return [doc for doc, _, _ in self._index.values()]
        return [
            doc
            for doc, _, _ in self._index.values()
            if self._filter_func(filters, doc)
        ]

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
    ) -> int:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :type documents: list[Document]
        :param policy: documents with the same ID count as duplicates.
            When duplicates are met, the store can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :type policy: DuplicatePolicy, optional

        :raises DuplicateDocumentError: Exception trigger on duplicate
            document if `policy=DuplicatePolicy.FAIL`

        :return: Number of documents written.
        :rtype: int
        """
        n_written = 0
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"Expected document type, got '{doc}' of type '{type(doc)}'."
                raise ValueError(msg)

            if doc.id in self._index.keys():
                if policy == DuplicatePolicy.SKIP:
                    continue
                elif policy == DuplicatePolicy.FAIL:
                    msg = f"Document with ID '{doc.id}' already exists in the store."
                    raise DuplicateDocumentError(msg)

                # Overwrite if exists; delete first to keep the statistics consistent
                logger.debug(
                    f"Document '{doc.id}' already exists in the store, overwriting."
                )
                self.delete_documents([doc.id])

            content = doc.content or ""
            if content == "" and isinstance(doc.dataframe, pd.DataFrame):
                content = doc.dataframe.astype(str).to_csv(index=False)

            tokens = self._tokenize(content)[0]

            self._index[doc.id] = (doc, Counter(tokens), len(tokens))
            self._freq_doc.update(set(tokens))
            self._avg_doc_len = (
                len(tokens) + self._avg_doc_len * len(self._index)
            ) / (len(self._index) + 1)

            logger.debug(f"Document '{doc.id}' written to store.")
            n_written += 1

        return n_written

    def delete_documents(self, document_ids: list[str]) -> int:
        """
        Deletes all documents with a matching document_ids.

        Fails with `MissingDocumentError` if no document with
        this id is present in the store.

        :param object_ids: the object_ids to delete
        :type object_ids: list[str]

        :raises MissingDocumentError: trigger on missing document.

        :return: Number of documents deleted.
        :rtype: int
        """
        n_removal = 0
        for doc_id in document_ids:
            try:
                _, freq, doc_len = self._index.pop(doc_id)
                self._freq_doc.subtract(Counter(freq.keys()))
                try:
                    self._avg_doc_len = (
                        self._avg_doc_len * (len(self._index) + 1) - doc_len
                    ) / len(self._index)
                except ZeroDivisionError:
                    self._avg_doc_len = 0

                logger.debug(f"Document '{doc_id}' deleted from store.")
                n_removal += 1
            except KeyError as exc:
                msg = f"Document with ID '{doc_id}' not found, cannot delete it."
                raise MissingDocumentError(msg) from exc

        return n_removal

    def to_dict(self) -> dict[str, Any]:
        """Serializes this store to a dictionary."""
        return default_to_dict(
            self,
            k=self.k,
            b=self.b,
            delta=self.delta * (self.k + 1.0),  # Because we scaled it on init
            sp_file=self._sp_file,
            n_grams=self._n_grams,
            haystack_filter_logic=self._haystack_filter_logic,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BetterBM25DocumentStore":
        """Deserializes the store from a dictionary."""
        return default_from_dict(cls, data)
