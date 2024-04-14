# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os
import pathlib
from collections import deque

import pandas as pd
import tqdm
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search import BaseSearch
from haystack import Document

from bbm25_haystack import BetterBM25DocumentStore

DIR_PROJ = pathlib.Path(__file__).resolve().parent.parent
DIR_BEIR_RAW = DIR_PROJ / "benchmarks" / "beir"

URL_BEIR = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)

DATASETS = [
    # General IR (in-domain)
    "msmarco",
    # Bio-medical IR
    "trec-covid",
    "nfcorpus",
    # Question answering
    "nq",
    "hotpotqa",
    "fiqa",
    # Citation prediction
    "scidocs",
    # Argument retrieval
    "arguana",
    "webis-touche2020",
    # Duplicate question retrieval
    "quora",
    "cqadupstack",
    # Fact checking
    "scifact",
    "fever",
    "climate-fever",
    # Entity retrieval
    "dbpedia-entity",
]

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


class BEIRWrapper(BaseSearch):
    """Wrapper for the BetterBM25DocumentStore to be compatible with BEIR."""

    def __init__(self, store: BetterBM25DocumentStore) -> None:
        self._store = store
        self._indexed = False

    def index(self, corpus: dict[str, dict[str, str]]) -> int:
        """Index the corpus for retrieval."""

        documents = []
        for idx, raw in tqdm.tqdm(corpus.items(), desc="Indexing corpus"):
            raw_title = raw.get("title", "")
            raw_text = raw.get("text", "")

            content = f"title: {raw_title}; text: {raw_text}"
            document = Document(idx, content=content)
            documents.append(document)

        self._indexed = True
        return self._store.write_documents(documents)

    def search(
        self,
        corpus: dict[str, dict[str]],
        queries: dict[str, str],
        top_k: int = 10,
        *args,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Search the corpus for relevant documents."""

        _ = args
        _ = kwargs

        if not self._indexed:
            self.index(corpus)

        results = {}
        for idx, qry in tqdm.tqdm(queries.items(), desc="Searching queries"):
            result = self._store._retrieval(qry, top_k=top_k)
            results[idx] = {doc.id: scr for doc, scr in result if doc.id != idx}
        return results


def download_dataset_from_beir(name: str) -> bool:
    """Download a dataset maintained by the UKP Lab."""

    if os.path.isdir(DIR_BEIR_RAW / name):
        logging.info(f"Dataset {name} already downloaded. Skipping...")
        return True

    try:
        logging.info(f"Downloading dataset {name} from BEIR to {DIR_BEIR_RAW}...")
        util.download_and_unzip(URL_BEIR.format(name=name), DIR_BEIR_RAW)
    except Exception as exc:
        logging.warn(f"Failed to download dataset {name} from BEIR: {exc}")
        return False

    logging.info(f"Dataset {name} downloaded successfully.")
    return True


def evaluate_retriever(args: argparse.Namespace) -> None:
    """Evaluate the retrieval performance of a query encoder over
    the BEIR benchmark."""

    queue = deque()  # [ local_save_dir_name... ]
    for name in args.datasets or DATASETS:
        download_dataset_from_beir(name)

        if name != "cqadupstack":
            queue.append(name)
            continue

        # Special handling for the CQADupStack dataset because the dataset has
        # subdirectories for each topic; so we need to flatten the directory.
        for sub_name in os.listdir(DIR_BEIR_RAW / "cqadupstack"):
            sub_name_alt = str(os.path.join("cqadupstack", sub_name))
            queue.append(sub_name_alt)

    records = []
    while queue:
        ds_name = queue.popleft()
        dir_raw = DIR_BEIR_RAW / ds_name

        store = BetterBM25DocumentStore(
            k=args.bm25_k1,
            b=args.bm25_b,
            delta=args.bm25_delta,
            sp_file=args.sp_file,
        )
        model = BEIRWrapper(store)
        retriever = EvaluateRetrieval(model)

        corpus, queries, qrels = GenericDataLoader(dir_raw).load(split=args.split)
        results = retriever.retrieve(corpus, queries)

        logging.info(f"Evaluating retriever over {ds_name}...")

        record = {}
        for metric in retriever.evaluate(qrels, results, k_values=args.k_values):
            record.update(metric)

        record.update(
            {
                "datetime": pd.Timestamp.now(),
                "dataset": ds_name.replace("/", "-"),
            }
        )
        record.update(store.to_dict().get("init_parameters"))
        records.append(record)

    records = pd.DataFrame(records)
    records.to_csv(args.output, index=False)


def get_args() -> argparse.Namespace:
    """Get command line arguments for evaluating retrieval performance."""

    parser = argparse.ArgumentParser(
        prog="benchmark_beir.py",
        description="Evaluate retrieval performance over the BEIR benchmark.",
        epilog="Email wangy49@seas.upenn.edu for questions.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=False,
        default=None,
        choices=DATASETS,
        help=(
            "Dataset names to evaluate on. All datasets will be used "
            "if not specified (default: None)"
        ),
    )
    parser.add_argument(
        "--bm25-k1",
        type=float,
        required=False,
        default=1.5,
        help="The BM25+ k1 parameter; default to 1.5",
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        required=False,
        help="The BM25+ b parameter; default to 0.75",
    )
    parser.add_argument(
        "--bm25-delta",
        type=float,
        default=1.0,
        required=False,
        help="The BM25+ delta parameter; default to 1.0",
    )
    parser.add_argument(
        "--sp-file",
        type=str,
        default=None,
        required=False,
        help="Path to the SentencePiece model file; default to None (LLaMA2)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        required=False,
        choices=["train", "dev", "test"],
        help="Dataset split to evaluate on (default: 'test')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="beir_evaluation_results.csv",
        help="Path to the evaluation result",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        required=False,
        default=[10, 100],
        help="Top-k values for evaluation (default: [10, 100])",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    evaluate_retriever(get_args())
