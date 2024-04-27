# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pytest
from haystack import DeserializationError, Pipeline
from haystack.dataclasses import Document
from haystack.testing.factory import document_store_class

from bbm25_haystack.bbm25_retriever import BetterBM25Retriever
from bbm25_haystack.bbm25_store import BetterBM25DocumentStore


@pytest.fixture()
def mock_docs():
    return [
        Document(content="Javascript is a popular programming language"),
        Document(content="Java is a popular programming language"),
        Document(content="Python is a popular programming language"),
        Document(content="Ruby is a popular programming language"),
        Document(content="PHP is a popular programming language"),
    ]


class TestRetriever:
    def test_init_default(self):
        retriever = BetterBM25Retriever(BetterBM25DocumentStore())
        assert retriever.filters is None
        assert retriever.top_k == 10

    def test_init_with_parameters(self):
        retriever = BetterBM25Retriever(
            BetterBM25DocumentStore(), filters={"name": "test.txt"}, top_k=5
        )
        assert retriever.filters == {"name": "test.txt"}
        assert retriever.top_k == 5

    def test_init_with_invalid_top_k_parameter(self):
        with pytest.raises(ValueError):
            BetterBM25Retriever(BetterBM25DocumentStore(), top_k=-2)

        with pytest.raises(TypeError):
            BetterBM25Retriever(BetterBM25DocumentStore(), top_k="2")

    def test_init_with_invalid_filters_parameter(self):
        with pytest.raises(TypeError):
            BetterBM25Retriever(BetterBM25DocumentStore(), filters="invalid")

    def test_to_dict(self):
        store_class = document_store_class(
            "MyFakeStore", bases=(BetterBM25DocumentStore,)
        )
        document_store = store_class()
        document_store.to_dict = lambda: {
            "type": "MyFakeStore",
            "init_parameters": {},
        }
        component = BetterBM25Retriever(document_store=document_store)

        data = component.to_dict()
        assert data == {
            "type": "bbm25_haystack.bbm25_retriever.BetterBM25Retriever",
            "init_parameters": {
                "document_store": {
                    "type": "MyFakeStore",
                    "init_parameters": {},
                },
                "filters": None,
                "top_k": 10,
                "set_score": True,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        ds = BetterBM25DocumentStore()
        serialized_ds = ds.to_dict()

        component = BetterBM25Retriever(
            document_store=BetterBM25DocumentStore(),
            filters={"name": "test.txt"},
            top_k=5,
            set_score=False,
        )
        data = component.to_dict()
        assert data == {
            "type": "bbm25_haystack.bbm25_retriever.BetterBM25Retriever",
            "init_parameters": {
                "document_store": serialized_ds,
                "filters": {"name": "test.txt"},
                "top_k": 5,
                "set_score": False,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "bbm25_haystack.bbm25_retriever.BetterBM25Retriever",
            "init_parameters": {
                "document_store": {
                    "type": "bbm25_haystack.bbm25_store.BetterBM25DocumentStore",
                    "init_parameters": {},
                },
                "filters": {"name": "test.txt"},
                "top_k": 5,
            },
        }
        component = BetterBM25Retriever.from_dict(data)
        assert isinstance(component.document_store, BetterBM25DocumentStore)
        assert component.filters == {"name": "test.txt"}
        assert component.top_k == 5

    def test_from_dict_without_docstore(self):
        data = {"type": "BetterBM25Retriever", "init_parameters": {}}
        with pytest.raises(
            DeserializationError,
            match="Missing 'document_store' in serialization data",
        ):
            BetterBM25Retriever.from_dict(data)

    def test_from_dict_without_docstore_type(self):
        data = {
            "type": "BetterBM25Retriever",
            "init_parameters": {"document_store": {"init_parameters": {}}},
        }
        with pytest.raises(
            DeserializationError,
            match="Missing 'type' in document store's serialization data",
        ):
            BetterBM25Retriever.from_dict(data)

    def test_from_dict_nonexisting_docstore(self):
        data = {
            "type": "bbm25_haystack.BetterBM25Retriever",
            "init_parameters": {
                "document_store": {
                    "type": "Nonexisting.Docstore",
                    "init_parameters": {},
                }
            },
        }
        with pytest.raises(DeserializationError):
            BetterBM25Retriever.from_dict(data)

    def test_retriever_valid_run(self, mock_docs):
        ds = BetterBM25DocumentStore()
        ds.write_documents(mock_docs)

        retriever = BetterBM25Retriever(ds, top_k=5)
        result = retriever.run(query="PHP")

        assert "documents" in result
        assert len(result["documents"]) == 5
        assert (
            result["documents"][0].content == "PHP is a popular programming language"
        )

    def test_invalid_run_wrong_store_type(self):
        store_class = document_store_class("SomeOtherDocumentStore")
        with pytest.raises(
            TypeError,
            match="'document_store' must be an instance of 'BetterBM25DocumentStore'",
        ):
            BetterBM25Retriever(store_class())

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query, query_result",
        [
            ("Javascript", "Javascript is a popular programming language"),
            ("Java", "Java is a popular programming language"),
        ],
    )
    def test_run_with_pipeline(self, mock_docs, query: str, query_result: str):
        ds = BetterBM25DocumentStore()
        ds.write_documents(mock_docs)
        retriever = BetterBM25Retriever(ds)

        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        result: dict[str, Any] = pipeline.run(data={"retriever": {"query": query}})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert results_docs[0].content == query_result

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query, query_result, top_k",
        [
            ("Javascript", "Javascript is a popular programming language", 1),
            ("Java", "Java is a popular programming language", 2),
            ("Ruby", "Ruby is a popular programming language", 3),
        ],
    )
    def test_run_with_pipeline_and_top_k(
        self, mock_docs, query: str, query_result: str, top_k: int
    ):
        ds = BetterBM25DocumentStore()
        ds.write_documents(mock_docs)
        retriever = BetterBM25Retriever(ds)

        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        result: dict[str, Any] = pipeline.run(
            data={"retriever": {"query": query, "top_k": top_k}}
        )

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert len(results_docs) == top_k
        assert results_docs[0].content == query_result
