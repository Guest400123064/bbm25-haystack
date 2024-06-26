# SPDX-FileCopyrightText: 2024-present Guest400123064 <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest
from haystack import Document
from haystack.document_stores.errors import (
    DuplicateDocumentError,
    MissingDocumentError,
)
from haystack.document_stores.types import (
    DocumentStore,
    DuplicatePolicy,
)
from haystack.errors import FilterError
from haystack.testing.document_store import (
    DocumentStoreBaseTests,
)

from bbm25_haystack.bbm25_store import BetterBM25DocumentStore


@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseTests):
    """Common test cases will be provided by `DocumentStoreBaseTests`."""

    @pytest.fixture
    def document_store(self) -> BetterBM25DocumentStore:
        return BetterBM25DocumentStore()

    @pytest.fixture
    def document_store_bbm25_filter(self) -> BetterBM25DocumentStore:
        return BetterBM25DocumentStore(haystack_filter_logic=False)

    def test_write_documents(self, document_store: DocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

        document_store.write_documents(
            [Document(id="1"), Document(id="2")], DuplicatePolicy.OVERWRITE
        )
        assert document_store.count_documents() == 2

    def test_delete_documents_empty_document_store(self, document_store):
        """
        This is different from the original implementation.

        One expects a MissingDocumentError to be raised when deleting a
        non-existing document, which is more intuitive.
        """
        with pytest.raises(MissingDocumentError):
            document_store.delete_documents(["non_existing_id"])

    def test_delete_documents_non_existing_document(self, document_store):
        """
        This is different from the original implementation.

        One expects a MissingDocumentError to be raised when deleting a
        non-existing document, which is more intuitive.
        """
        document_store.write_documents([Document(id="42")])
        with pytest.raises(MissingDocumentError):
            document_store.delete_documents(["non_existing_id"])

        assert document_store.count_documents() == 1

    def test_bm25_retrieval(self, document_store):
        docs = [
            Document(content="Hello world"),
            Document(content="Haystack supports multiple languages"),
        ]
        document_store.write_documents(docs)

        results = document_store._retrieval(query="What languages?", top_k=1)

        assert len(results) == 1
        assert results[0][0].content == "Haystack supports multiple languages"

    # Override a few filter test cases to account for new comparison logic
    # Specifically, we alter the expected behavior when comparison involves
    # None, DataFrame, and Iterables.
    def test_comparison_equal_with_none_bbm25_filter(
        self, document_store_bbm25_filter, filterable_docs
    ):
        document_store_bbm25_filter.write_documents(filterable_docs)
        result = document_store_bbm25_filter.filter_documents(
            filters={"field": "meta.number", "operator": "==", "value": None}
        )
        self.assert_documents_are_equal(result, [])

    def test_comparison_not_equal_with_none_bbm25_filter(
        self, document_store_bbm25_filter, filterable_docs
    ):
        document_store_bbm25_filter.write_documents(filterable_docs)
        result = document_store_bbm25_filter.filter_documents(
            filters={"field": "meta.number", "operator": "!=", "value": None}
        )
        self.assert_documents_are_equal(result, [])

    def test_comparison_not_equal_bbm25_filter(
        self, document_store_bbm25_filter, filterable_docs
    ):
        """Comparison with missing values will always return False.
        So the ground truth is that we should only return documents
        with a non-missing value."""
        document_store_bbm25_filter.write_documents(filterable_docs)
        result = document_store_bbm25_filter.filter_documents(
            {"field": "meta.number", "operator": "!=", "value": 100}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("number") != 100 and "number" in d.meta
            ],
        )

    def test_comparison_not_in_bbm25_filter(
        self, document_store_bbm25_filter, filterable_docs
    ):
        """Similar to the test above."""
        document_store_bbm25_filter.write_documents(filterable_docs)
        result = document_store_bbm25_filter.filter_documents(
            {"field": "meta.number", "operator": "not in", "value": [9, 10]}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("number") not in [9, 10] and "number" in d.meta
            ],
        )

    def test_comparison_equal_with_dataframe_bbm25_filter(
        self, document_store_bbm25_filter, filterable_docs
    ):
        document_store_bbm25_filter.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            _ = document_store_bbm25_filter.filter_documents(
                filters={
                    "field": "dataframe",
                    "operator": "==",
                    "value": pd.DataFrame([1]),
                }
            )

    def test_comparison_not_equal_with_dataframe_bbm25_filter(
        self, document_store_bbm25_filter, filterable_docs
    ):
        document_store_bbm25_filter.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            _ = document_store_bbm25_filter.filter_documents(
                filters={
                    "field": "dataframe",
                    "operator": "==",
                    "value": pd.DataFrame([1]),
                }
            )
