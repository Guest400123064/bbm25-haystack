# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
import heapq
import logging
import math
import os
from collections import Counter
from typing import Any, Callable, Final, Optional, TypeAlias, Union

from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import (
    DuplicateDocumentError,
    MissingDocumentError,
)
from haystack.document_stores.types import DuplicatePolicy
from sentencepiece import SentencePieceProcessor  # type: ignore

from bbm25_haystack.filters import apply_filters_to_document

logger = logging.getLogger(__name__)


DEFAULT_SP_MODEL: Final = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "default.model"
)

IndexType: TypeAlias = dict[
    str, tuple[Document, dict[str, int], int, Callable[[float], float]]
]


class BetterBM25DocumentStore:
    """
    An in-memory document store intended to improve the default BM25 document
    store shipped with Haystack.
    """

    def __init__(
        self,
        *,
        k: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        sp_file: Optional[str] = None,
    ):
        """
        Creates a new BetterBM25DocumentStore instance.

        An in-memory document store intended to improve the default BM25 document
        store shipped with Haystack. The default store recompute the index for the
        entire document store for every in-coming query, which is significantly
        inefficient. This store aims to improve the efficiency by pre-computing
        the index for all documents in the store and only do incremental updates
        when new documents are added or removed. Further, it leverages a
        SentencePiece model to tokenize the input text to allow more flexible
        and dynamic tokenization adapted to domain-specific text.

        :param k: the k1 parameter in BM25+ formula.
        :type k: float, optional
        :param b: the b parameter in BM25+ formula.
        :type b: float, optional
        :param delta: the delta parameter in BM25+ formula.
        :type delta: float, optional
        :param sp_file: the SentencePiece model file to use for tokenization.
        :type sp_file: Optional[str], optional
        """
        self.k = k
        self.b = b

        # Adjust the delta value so that we can bring the `(k1 + 1)`
        # term out of the 'term frequency' term in BM25+ formula and
        # delete it; this will not affect the ranking
        self.delta = delta / (self.k + 1.0)

        self._sp_file = sp_file
        self._sp_inst = SentencePieceProcessor(
            model_file=(self._sp_file or DEFAULT_SP_MODEL)
        )

        self._avg_doc_len: float = 0.0
        self._freq_doc: Counter = Counter()

        self._index: IndexType = {}

    def _tokenize(self, texts: Union[str, list[str]]) -> list[list[str]]:
        """
        Tokenize input text using SentencePiece model.

        The input text can either be a single string or a list of strings,
        such as a single user query or a group of raw document.

        :param texts: the input text to tokenize.
        :type texts: Union[str, list[str]]

        :return: the tokenized text.
        :rtype: list[list[str]]
        """
        if isinstance(texts, str):
            texts = [texts]
        return self._sp_inst.encode(texts, out_type=str)

    def _compute_idf(self, tokens: list[str]) -> dict[str, float]:
        """
        Calculate the inverse document frequency for each token.

        :param tokens: the tokens to calculate the IDF for.
        :type tokens: list[str]

        :return: the IDF for each token.
        :rtype: dict[str, float]
        """
        n = lambda t: self._freq_doc.get(t, 0)
        idf = {
            t: math.log(1 + (len(self._index) - n(t) + 0.5) / (n(t) + 0.5))
            for t in tokens
        }
        return idf

    def _compute_all_bm25plus(
        self,
        idf: dict[str, float],
        documents: list[Document],
    ) -> list[tuple[Document, float]]:
        """
        Calculate the BM25+ score for all documents in this index.

        :param idf: the IDF for each token.
        :type idf: dict[str, float]
        :param documents: the pool of documents to calculate the BM25+ score for.
        :type documents: list[Document]

        :return: the BM25+ scores for all documents.
        :rtype: list[tuple[Document, float]]
        """
        f = lambda t, d: d.get(t, 0.0)
        store = {doc.id: self._index[doc.id] for doc in documents}

        scores = [
            (
                doc,
                sum(
                    idf[token]
                    * (
                        f(token, freq) / (f(token, freq) + norm(self._avg_doc_len))
                        + self.delta
                    )
                    for token in idf.keys()
                ),
            )
            for doc, freq, _, norm in store.values()
        ]
        return scores

    def _retrieval(
        self,
        query: str,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Retrieve documents from the store using the given query.

        :param query: the query to search for.
        :type query: str
        :param filters: the filters to apply to the document list.
        :type filters: Optional[dict[str, Any]]
        :param top_k: the number of documents to return.
        :type top_k: int

        :return: the top-k documents that match the query.
        :rtype: list[Document]
        """
        documents = self.filter_documents(filters)
        if not documents:
            return []

        idf = self._compute_idf(self._tokenize(query)[0])
        sim = self._compute_all_bm25plus(idf, documents)

        key = lambda x: x[1]
        if top_k is None:
            top = sorted(sim, key=key, reverse=True)
        else:
            top = heapq.nlargest(top_k, sim, key=key)

        return next(zip(*top))  # type: ignore

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
        return [
            doc[0]
            for doc in self._index.values()
            if apply_filters_to_document(filters, doc[0])
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
        :param policy: documents with the same ID count as duplicates. When
            duplicates are met, the store can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :type policy: DuplicatePolicy, optional

        :raises DuplicateDocumentError: Exception trigger on duplicate document if
            `policy=DuplicatePolicy.FAIL`

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

            tokens = self._tokenize(doc.content or "")[0]
            doc_len = len(tokens)

            self._freq_doc.update(set(tokens))
            self._avg_doc_len = (len(tokens) + self._avg_doc_len * len(self._index)) / (
                len(self._index) + 1
            )

            def tf(adl: float, dl: int = doc_len) -> float:
                return self.k * (1 - self.b + self.b * dl / adl)

            self._index[doc.id] = (doc, Counter(tokens), doc_len, tf)

            logger.debug(f"Document '{doc.id}' written to store.")
            n_written += 1

        return n_written

    def delete_documents(self, document_ids: list[str]) -> int:
        """
        Deletes all documents with a matching document_ids from the document store.

        Fails with `MissingDocumentError` if no document with this id is present in
        the store.

        :param object_ids: the object_ids to delete
        :type object_ids: list[str]

        :raises MissingDocumentError: Exception trigger on missing document.

        :return: Number of documents deleted.
        :rtype: int
        """
        n_removal = 0

        for doc_id in document_ids:
            try:
                _, freq, doc_len, _ = self._index.pop(doc_id)

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
            delta=self.delta,
            sp_file=self._sp_file,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BetterBM25DocumentStore":
        """Deserializes the store from a dictionary."""
        return default_from_dict(cls, data)
