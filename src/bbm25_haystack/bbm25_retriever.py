# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

from haystack import (
    DeserializationError,
    Document,
    component,
    default_from_dict,
    default_to_dict,
)

from bbm25_haystack.bbm25_store import BetterBM25DocumentStore


def _validate_search_params(filters: Optional[dict[str, Any]], top_k: int) -> None:
    """
    Validate the search parameters.

    :param filters: A dictionary with filters to narrow down the search space
        (default is None).
    :type filters: Optional[dict[str, Any]]
    :param top_k: The maximum number of documents to retrieve (default is 10).
    :type top_k: int

    :raises ValueError: If the specified top_k is not > 0.
    :raises TypeError: If filters is not a dictionary.
    """
    if not isinstance(top_k, int):
        msg = f"top_k must be an integer; got {type(top_k)} instead"
        raise TypeError(msg)

    if top_k <= 0:
        msg = f"top_k must be > 0; got {top_k} instead"
        raise ValueError(msg)

    if filters is not None and (not isinstance(filters, dict)):
        msg = f"filters must be a dictionary; got {type(filters)} instead"
        raise TypeError(msg)


@component
class BetterBM25Retriever:
    """
    A component for retrieving documents from an BetterBM25DocumentStore.
    """

    def __init__(
        self,
        document_store: BetterBM25DocumentStore,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        set_score: bool = True,
    ) -> None:
        """
        Create an BetterBM25Retriever component.

        :param document_store: A Document Store object used to
            retrieve documents
        :type document_store: BetterBM25DocumentStore
        :param filters: A dictionary with filters to narrow down the
            search space (default is None).
        :type filters: Optional[dict[str, Any]]
        :param top_k: The maximum number of documents to retrieve
            (default is 10).
        :type top_k: int
        :param set_score: Whether to set the similarity scores
            to retrieved documents (default is True).
        :type set_score: bool

        :raises ValueError: If the specified top_k is not > 0.
        """
        _validate_search_params(filters, top_k)

        self.filters = filters
        self.top_k = top_k
        self.set_score = set_score

        if not isinstance(document_store, BetterBM25DocumentStore):
            msg = "document_store must be an instance of BetterBM25DocumentStore"
            raise TypeError(msg)
        self.document_store = document_store

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, list[Document]]:
        """
        Run the Retriever on the given query.

        This method always return copies of the documents
        retrieved from the document store.

        :param query: The query to run the Retriever on.
        :type query: str
        :param filters: A dictionary with filters to narrow
            down the search space (default is None).
        :type filters: Optional[dict[str, Any]]
        :param top_k: The maximum number of documents to
            retrieve (default is None).

        :return: The retrieved documents.
        """
        filters = filters or self.filters
        top_k = top_k or self.top_k

        _validate_search_params(filters, top_k)

        sim = self.document_store._retrieval(query, filters=filters, top_k=top_k)

        ret = []
        for doc, score in sim:
            data = doc.to_dict()
            if self.set_score:
                data["score"] = score
            ret.append(Document.from_dict(data))

        return {"documents": ret}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :return: dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            document_store=self.document_store.to_dict(),
            set_score=self.set_score,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BetterBM25Retriever":
        """
        Deserializes the component from a dictionary.

        :param data: dictionary to deserialize from.
        :returns: deserialized component.
        """
        doc_store_params = data["init_parameters"].get("document_store")
        if doc_store_params is None:
            msg = "Missing 'document_store' in serialization data"
            raise DeserializationError(msg)

        if doc_store_params.get("type") is None:
            msg = "Missing 'type' in document store's serialization data"
            raise DeserializationError(msg)

        data["init_parameters"]["document_store"] = (
            BetterBM25DocumentStore.from_dict(doc_store_params)
        )
        return default_from_dict(cls, data)
