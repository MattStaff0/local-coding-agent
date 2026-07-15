"""Optional cross-encoder reranking for retrieval precision.

Order matters: hybrid retrieval (vector + BM25 + RRF) casts a wide net for
recall; a cross-encoder then rescores question/chunk pairs directly for
precision before the final top-k cut. Off by default — turn it on with
RAG_RERANKER=cross-encoder and prove it helps with eval_retrieval.py
before/after, numbers not vibes.

Requires `pip install sentence-transformers` (not in requirements.txt on
purpose: it drags in torch, and the feature is opt-in).
"""
import os

RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

_model = None
_warned = False


def enabled() -> bool:
    """Reranking is opt-in so default retrieval (and eval baselines) never change."""
    return os.getenv("RAG_RERANKER", "").strip().lower() in {
        "1", "true", "on", "cross-encoder",
    }


def _predict_fn():
    """Load the cross-encoder once per process; raises if not installed."""
    global _model

    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(RERANK_MODEL)

    return _model.predict


def rerank(
    question: str,
    ids: list[str],
    documents: list[str],
    n_results: int,
) -> list[str]:
    """Order candidate ids by cross-encoder relevance; never break retrieval.

    Any failure (model not installed, download blocked) keeps the fusion
    order — a missing reranker degrades precision, not availability.
    """
    global _warned

    try:
        predict = _predict_fn()
        scores = predict([(question, document) for document in documents])
    except Exception as error:
        if not _warned:
            print(
                f"(reranker unavailable: {type(error).__name__}: {error} "
                "— keeping hybrid fusion order)"
            )
            _warned = True
        return ids[:n_results]

    ranked = sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True)
    return [item_id for item_id, _ in ranked][:n_results]
