# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from bbm25_haystack.bbm25_store import BetterBM25DocumentStore


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
    ):
        """
        Create an BetterBM25Retriever component.

        :param document_store: A Document Store object used to retrieve documents
        :type document_store: BetterBM25DocumentStore
        :param filters: A dictionary with filters to narrow down the search space
            (default is None).
        :type filters: Optional[dict[str, Any]]
        :param top_k: The maximum number of documents to retrieve (default is 10).
        :type top_k: int

        :raises ValueError: If the specified top_k is not > 0.
        """
        self.filters = filters
        self.top_k = top_k
        self.document_store = document_store

    def run(
        self,
        query: str,
        *,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, list[Document]]:
        """
        Run the Retriever on the given query.

        :param query: The query to run the Retriever on.
        :type query: str
        :param filters: A dictionary with filters to narrow down the search space
            (default is None).
        :type filters: Optional[dict[str, Any]]
        :param top_k: The maximum number of documents to retrieve (default is None).

        :return: The retrieved documents.
        """
        filters = filters or self.filters
        top_k = top_k or self.top_k

        docs = self.document_store._retrieval(query, filters=filters, top_k=top_k)
        return {"documents": docs}

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
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BetterBM25Retriever":
        """
        Deserializes the component from a dictionary.

        :param data: dictionary to deserialize from.
        :returns: deserialized component.
        """
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = BetterBM25DocumentStore.from_dict(
            doc_store_params
        )
        return default_from_dict(cls, data)
