# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Tuple, List, Union, Optional, Any, Final
from collections import Counter

import os
import logging

import math
import heapq

from sentencepiece import SentencePieceProcessor

from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import (
    DuplicateDocumentError,
    MissingDocumentError,
)
from haystack.document_stores.types import DuplicatePolicy


logger = logging.getLogger(__name__)


DIR_THIS: Final = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SP_MODEL: Final = os.path.join(DIR_THIS, "default.model")


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

        self._avg_doc_len = 0
        self._freq_doc = Counter()

        self._index = {}

    def _tokenize(self, texts: Union[str, List[str]]) -> List[List[str]]:
        """
        Tokenize input text using SentencePiece model.

        The input text can either be a single string or a list of strings,
        such as a single user query or a group of raw document.

        :param texts: the input text to tokenize.
        :type texts: Union[str, List[str]]

        :return: the tokenized text.
        :rtype: List[List[str]]
        """
        if isinstance(texts, str):
            texts = [texts]
        return self._sp_inst.encode(texts, out_type=str)

    def _compute_idf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate the inverse document frequency for each token.

        :param tokens: the tokens to calculate the IDF for.
        :type tokens: List[str]

        :return: the IDF for each token.
        :rtype: Dict[str, float]
        """
        n = lambda t: self._freq_doc.get(t, 0)
        idf = {
            t: math.log(1 + (len(self._index) - n(t) + 0.5) / (n(t) + 0.5))
            for t in tokens
        }
        return idf

    def _compute_all_bm25plus(
        self,
        idf: Dict[str, float],
        doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Calculate the BM25+ score for all documents in this index.

        :param idf: the IDF for each token.
        :type idf: Dict[str, float]
        :param doc_ids: the document IDs to calculate the BM25+ score for.
        :type doc_ids: Optional[List[str]

        :return: the BM25+ scores for all documents.
        :rtype: List[float]
        """
        f = lambda t, d: d.get(t, 0)
        store = (
            self._index
            if doc_ids is None
            else {k: self._index[k] for k in doc_ids}
        )

        scores = [
            (
                doc, sum(
                    idf[token] * (
                        f(token, freq)
                        / (f(token, freq) + norm(self._avg_doc_len))
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
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Retrieve documents from the store using the given query.

        :param query: the query to search for.
        :type query: str
        :param filters: the filters to apply to the document list.
        :type filters: Optional[Dict[str, Any]]
        :param top_k: the number of documents to return.
        :type top_k: int

        :return: the top-k documents that match the query.
        :rtype: List[Document]
        """
        doc_ids = self.filter_documents(filters)

        idf = self._compute_idf(self._tokenize(query)[0])
        sim = self._compute_all_bm25plus(idf, doc_ids)

        key = lambda x: x[1]
        if top_k is None:
            top = sorted(sim, key=key, reverse=True)
        else:
            top = heapq.nlargest(top_k, sim, key=key)

        return [doc for doc, _ in top]

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :return: the number of documents in the store.
        :rtype: int
        """
        return len(self._index)

    def filter_documents(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Filter documents in the store using the given filters.

        :param filters: the filters to apply to the document list.
        :type filters: Optional[Dict[str, Any]]

        :return: the list of documents that match the given filters.
        :rtype: List[Document]
        """
        return None

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :type documents: List[Document]
        :param policy: documents with the same ID count as duplicates. When duplicates are met,
            the store can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :type policy: DuplicatePolicy, optional

        :raises DuplicateDocumentError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`

        :return: Number of documents written.
        :rtype: int
        """
        n_written = 0

        for doc in documents:
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

            tokens = self._tokenize(doc.content)[0]

            self._freq_doc.update(set(tokens))
            self._avg_doc_len = (
                len(tokens) + self._avg_doc_len * len(self._index)
            ) / (len(self._index) + 1)

            self._index[doc.id] = (
                doc,
                Counter(tokens),
                len(tokens),
                lambda adl: self.k * (1 - self.b + self.b * len(tokens) / adl),
            )

            logger.debug(f"Document '{doc.id}' written to store.")
            n_written += 1

        return n_written

    def delete_documents(self, document_ids: List[str]) -> int:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param object_ids: the object_ids to delete
        :type object_ids: List[str]

        :raises MissingDocumentError: Exception trigger on missing document.

        :return: Number of documents deleted.
        :rtype: int
        """
        n_removal = 0

        for doc_id in document_ids:
            try:
                doc, freq, doc_len, _ = self._index.pop(doc_id)
                assert (
                    doc.id == doc_id
                ), f"Unexpected document ID mismatch: {doc_id} != {doc.id}"
            except KeyError as _:
                msg = (
                    f"Document with ID '{doc_id}' not found, cannot delete it."
                )
                raise MissingDocumentError(msg)

            self._freq_doc.subtract(Counter(freq.keys()))
            self._avg_doc_len = (
                self._avg_doc_len * (len(self._index) + 1) - doc_len
            ) / len(self._index)

            logger.debug(f"Document '{doc_id}' deleted from store.")
            n_removal += 1

        return n_removal

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this store to a dictionary."""
        return default_to_dict(
            self,
            k=self.k,
            b=self.b,
            delta=self.delta,
            sp_file=self._sp_file,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BetterBM25DocumentStore":
        """Deserializes the store from a dictionary."""
        return default_from_dict(cls, data)
