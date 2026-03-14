import json
import os
import re
from typing import Dict, List, Tuple
from math import sqrt

from langchain_community.embeddings import HuggingFaceEmbeddings

import config


_SEMANTIC_EMBEDDER = None


def load_golden_dataset(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            source = str(item.get("source", "")).strip()
            if question and answer:
                rows.append({
                    "question": question,
                    "answer": answer,
                    "source": source,
                })
    return rows


def _normalize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction)
    ref_tokens = _normalize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    common = 0
    for token, p_count in pred_counts.items():
        if token in ref_counts:
            common += min(p_count, ref_counts[token])

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _get_semantic_embedder():
    global _SEMANTIC_EMBEDDER
    if _SEMANTIC_EMBEDDER is None:
        _SEMANTIC_EMBEDDER = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )
    return _SEMANTIC_EMBEDDER


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _semantic_similarity(prediction: str, reference: str) -> float:
    if not prediction.strip() or not reference.strip():
        return 0.0
    try:
        embedder = _get_semantic_embedder()
        vectors = embedder.embed_documents([prediction, reference])
        return _cosine_similarity(vectors[0], vectors[1])
    except Exception:
        return 0.0


def evaluate_dataset(doc_processor, tutor_engine, dataset: List[Dict[str, str]], k_values: Tuple[int, ...] = (1, 3, 5)) -> Dict:
    if not dataset:
        return {
            "num_samples": 0,
            "hit_rates": {f"Hit@{k}": 0.0 for k in k_values},
            "mrr": 0.0,
            "avg_f1": 0.0,
            "avg_semantic_similarity": 0.0,
            "rows": [],
        }

    hit_counts = {k: 0 for k in k_values}
    mrr_total = 0.0
    f1_total = 0.0
    semantic_total = 0.0
    rows = []

    max_k = max(k_values)

    for sample in dataset:
        question = sample["question"]
        expected_answer = sample["answer"]
        expected_source = sample.get("source", "")

        docs = doc_processor.retrieve_docs(question, k=max_k)
        retrieved_sources = [str(d.metadata.get("source", "")) for d in docs]

        rank = None
        if expected_source:
            for idx, src in enumerate(retrieved_sources, start=1):
                if src == expected_source:
                    rank = idx
                    break

            for k in k_values:
                if rank is not None and rank <= k:
                    hit_counts[k] += 1
            if rank is not None:
                mrr_total += 1.0 / rank

        predicted_answer = tutor_engine.get_response(question, record_memory=False)
        f1_score = _token_f1(predicted_answer, expected_answer)
        semantic_sim = _semantic_similarity(predicted_answer, expected_answer)
        f1_total += f1_score
        semantic_total += semantic_sim

        rows.append({
            "question": question,
            "expected_source": expected_source,
            "retrieved_top1": retrieved_sources[0] if retrieved_sources else "",
            "rank_of_expected_source": rank if rank is not None else "not_found",
            "answer_f1": round(f1_score, 3),
            "semantic_similarity": round(semantic_sim, 3),
        })

    n = len(dataset)
    hit_rates = {f"Hit@{k}": round(hit_counts[k] / n, 3) for k in k_values}
    mrr = round(mrr_total / n, 3)
    avg_f1 = round(f1_total / n, 3)
    avg_semantic_similarity = round(semantic_total / n, 3)

    return {
        "num_samples": n,
        "hit_rates": hit_rates,
        "mrr": mrr,
        "avg_f1": avg_f1,
        "avg_semantic_similarity": avg_semantic_similarity,
        "rows": rows,
    }
