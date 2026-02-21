#!/usr/bin/env python3
"""
L1: Concept-Level Router Accuracy Test Bench

Compares sentence-embedding k-NN classification against the nvidia_orchestrator-8b
LLM router for intent routing in nanobot trio mode.

Supports two embedding backends:
  1. sentence-transformers (local, default): all-MiniLM-L6-v2 (~80MB)
  2. LM Studio /v1/embeddings (remote): any loaded embedding model

Usage:
  pip install sentence-transformers numpy
  python test_bench.py                          # local sentence-transformers
  python test_bench.py --backend lmstudio       # LM Studio embeddings
  python test_bench.py --backend lmstudio --host 192.168.1.22:1234
"""

import json
import time
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    text: str
    vector: np.ndarray
    latency_ms: float

@dataclass
class ClassificationResult:
    query: str
    predicted_action: str
    predicted_target: str | None
    confidence: float
    expected_action: str
    expected_target: str | None
    correct_action: bool
    correct_target: bool
    latency_ms: float
    top_k_labels: list[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class SentenceTransformerBackend:
    """Local sentence-transformers (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        # Warmup
        self.model.encode(["warmup"], show_progress_bar=False)
        print(f"Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        results = []
        for text in texts:
            t0 = time.perf_counter()
            vec = self.model.encode([text], show_progress_bar=False)[0]
            dt = (time.perf_counter() - t0) * 1000
            results.append(EmbeddingResult(text=text, vector=vec, latency_ms=dt))
        return results

    def embed_batch(self, texts: list[str]) -> tuple[np.ndarray, float]:
        """Embed a batch, return (matrix, total_ms)."""
        t0 = time.perf_counter()
        vecs = self.model.encode(texts, show_progress_bar=False)
        dt = (time.perf_counter() - t0) * 1000
        return np.array(vecs), dt


class LMStudioBackend:
    """Remote LM Studio /v1/embeddings endpoint."""

    def __init__(self, host: str = "192.168.1.22:1234", model: str | None = None):
        import urllib.request
        self.base_url = f"http://{host}/v1"
        self.model = model or self._detect_model()
        print(f"Using LM Studio embeddings: {self.model} @ {host}")

    def _detect_model(self) -> str:
        import urllib.request, json as _json
        try:
            req = urllib.request.urlopen(f"{self.base_url}/models", timeout=5)
            data = _json.loads(req.read())
            for m in data.get("data", []):
                mid = m.get("id", "")
                if "embed" in mid.lower() or "nomic" in mid.lower():
                    return mid
            # fallback to first model
            if data.get("data"):
                return data["data"][0]["id"]
        except Exception as e:
            print(f"Warning: Could not detect model: {e}")
        return "nomic-embed-text-v1.5"

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        import urllib.request, json as _json
        payload = _json.dumps({"input": texts, "model": self.model}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = _json.loads(resp.read())
        # Sort by index to preserve order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [d["embedding"] for d in sorted_data]

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        results = []
        for text in texts:
            t0 = time.perf_counter()
            vecs = self._call_api([text])
            dt = (time.perf_counter() - t0) * 1000
            results.append(EmbeddingResult(text=text, vector=np.array(vecs[0]), latency_ms=dt))
        return results

    def embed_batch(self, texts: list[str]) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        vecs = self._call_api(texts)
        dt = (time.perf_counter() - t0) * 1000
        return np.array(vecs), dt


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ConceptRouter:
    """Concept router using cosine similarity. Supports k-NN and centroid strategies."""

    def __init__(self, backend, reference_path: str, k: int = 5,
                 strategy: str = "centroid"):
        self.backend = backend
        self.k = k
        self.strategy = strategy  # "knn" or "centroid"
        self.labels: list[str] = []       # action label per reference example
        self.ref_texts: list[str] = []
        self.ref_matrix: np.ndarray | None = None
        # Centroid data
        self.action_centroids: dict[str, np.ndarray] = {}    # action -> mean vector
        self.subtype_centroids: dict[str, np.ndarray] = {}   # label -> mean vector

        self._load_reference_set(reference_path)

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return matrix / norms

    def _load_reference_set(self, path: str):
        with open(path) as f:
            data = json.load(f)

        texts = []
        labels = []
        for label, examples in data.items():
            if label.startswith("_"):
                continue
            for ex in examples:
                texts.append(ex)
                labels.append(label)

        print(f"Embedding {len(texts)} reference examples ...")
        self.ref_texts = texts
        self.labels = labels
        self.ref_matrix, embed_ms = self.backend.embed_batch(texts)
        self.ref_matrix = self._normalize(self.ref_matrix)

        # Build centroids per action and per subtype
        action_groups: dict[str, list[int]] = defaultdict(list)
        subtype_groups: dict[str, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            action = self._action_of(label)
            action_groups[action].append(i)
            subtype_groups[label].append(i)

        for action, indices in action_groups.items():
            centroid = self.ref_matrix[indices].mean(axis=0)
            self.action_centroids[action] = self._normalize(centroid.reshape(1, -1))[0]

        for label, indices in subtype_groups.items():
            centroid = self.ref_matrix[indices].mean(axis=0)
            self.subtype_centroids[label] = self._normalize(centroid.reshape(1, -1))[0]

        print(f"Reference set embedded in {embed_ms:.0f}ms ({len(texts)} examples, "
              f"dim={self.ref_matrix.shape[1]})")
        print(f"Strategy: {self.strategy} | "
              f"Action centroids: {list(action_groups.keys())} | "
              f"Counts: {{{', '.join(f'{k}: {len(v)}' for k, v in sorted(action_groups.items()))}}}")

    @staticmethod
    def _action_of(label: str) -> str:
        """Extract action from label: 'tool:read_file' -> 'tool'."""
        return label.split(":")[0] if ":" in label else label

    @staticmethod
    def _target_of(label: str) -> str | None:
        """Extract target from label: 'tool:read_file' -> 'read_file'."""
        return label.split(":", 1)[1] if ":" in label else None

    def classify(self, query: str) -> ClassificationResult:
        """Classify a query. Delegates to centroid or k-NN strategy."""
        if self.strategy == "centroid":
            return self._classify_centroid(query)
        return self._classify_knn(query)

    def _classify_centroid(self, query: str) -> ClassificationResult:
        """Classify by nearest action centroid. Immune to class imbalance."""
        t0 = time.perf_counter()

        results = self.backend.embed([query])
        q_vec = self._normalize(results[0].vector.reshape(1, -1))[0]

        # Cosine similarity to each action centroid
        action_sims = {}
        for action, centroid in self.action_centroids.items():
            action_sims[action] = float(np.dot(q_vec, centroid))

        predicted_action = max(action_sims, key=action_sims.get)
        total = sum(max(0, s) for s in action_sims.values())
        confidence = max(0, action_sims[predicted_action]) / total if total > 0 else 0

        # Find best subtype target within the predicted action
        predicted_target = None
        subtype_sims = {}
        for label, centroid in self.subtype_centroids.items():
            if self._action_of(label) == predicted_action and self._target_of(label):
                subtype_sims[label] = float(np.dot(q_vec, centroid))
        if subtype_sims:
            best_label = max(subtype_sims, key=subtype_sims.get)
            predicted_target = self._target_of(best_label)

        dt = (time.perf_counter() - t0) * 1000

        # Debug: top-k action similarities
        sorted_actions = sorted(action_sims.items(), key=lambda x: x[1], reverse=True)
        top_k_debug = [f"{a} ({s:.3f})" for a, s in sorted_actions]

        return ClassificationResult(
            query=query,
            predicted_action=predicted_action,
            predicted_target=predicted_target,
            confidence=confidence,
            expected_action="",
            expected_target=None,
            correct_action=False,
            correct_target=False,
            latency_ms=dt,
            top_k_labels=top_k_debug,
        )

    def _classify_knn(self, query: str) -> ClassificationResult:
        """Classify using k-NN with action-level weighted voting."""
        t0 = time.perf_counter()

        results = self.backend.embed([query])
        q_vec = self._normalize(results[0].vector.reshape(1, -1))[0]

        similarities = self.ref_matrix @ q_vec

        top_k_idx = np.argsort(similarities)[-self.k:][::-1]
        top_k_labels = [self.labels[i] for i in top_k_idx]
        top_k_sims = [float(similarities[i]) for i in top_k_idx]

        action_weights: dict[str, float] = defaultdict(float)
        target_votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for label, sim in zip(top_k_labels, top_k_sims):
            action = self._action_of(label)
            target = self._target_of(label)
            action_weights[action] += sim
            if target:
                target_votes[action][target] += sim

        predicted_action = max(action_weights, key=action_weights.get)
        total_weight = sum(action_weights.values())
        confidence = action_weights[predicted_action] / total_weight if total_weight > 0 else 0

        predicted_target = None
        if predicted_action in target_votes:
            predicted_target = max(target_votes[predicted_action],
                                   key=target_votes[predicted_action].get)

        dt = (time.perf_counter() - t0) * 1000

        return ClassificationResult(
            query=query,
            predicted_action=predicted_action,
            predicted_target=predicted_target,
            confidence=confidence,
            expected_action="",
            expected_target=None,
            correct_action=False,
            correct_target=False,
            latency_ms=dt,
            top_k_labels=[f"{l} ({s:.3f})" for l, s in zip(top_k_labels, top_k_sims)],
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(router: ConceptRouter, test_path: str) -> list[ClassificationResult]:
    with open(test_path) as f:
        data = json.load(f)

    results = []
    for tc in data["test_cases"]:
        result = router.classify(tc["query"])
        result.expected_action = tc["expected_action"]
        result.expected_target = tc.get("expected_target")
        result.correct_action = result.predicted_action == tc["expected_action"]
        result.correct_target = (
            tc.get("expected_target") is None  # no target to check
            or result.predicted_target == tc.get("expected_target")
        )
        results.append(result)

    return results


def print_report(results: list[ClassificationResult]):
    total = len(results)
    correct_action = sum(1 for r in results if r.correct_action)
    correct_both = sum(1 for r in results if r.correct_action and r.correct_target)
    latencies = [r.latency_ms for r in results]

    print("\n" + "=" * 80)
    print("L1 CONCEPT ROUTER ACCURACY REPORT")
    print("=" * 80)

    # Overall
    print(f"\nAction accuracy: {correct_action}/{total} ({100*correct_action/total:.1f}%)")
    print(f"Action+target accuracy: {correct_both}/{total} ({100*correct_both/total:.1f}%)")
    print(f"Latency: mean={np.mean(latencies):.1f}ms, "
          f"median={np.median(latencies):.1f}ms, "
          f"p95={np.percentile(latencies, 95):.1f}ms, "
          f"max={np.max(latencies):.1f}ms")

    # Per-category breakdown
    categories: dict[str, list[ClassificationResult]] = defaultdict(list)
    for r in results:
        categories[r.expected_action].append(r)

    print(f"\n{'Action':<12} {'Correct':>8} {'Total':>6} {'Precision':>10}")
    print("-" * 40)
    for action in sorted(categories):
        group = categories[action]
        c = sum(1 for r in group if r.correct_action)
        print(f"{action:<12} {c:>8} {len(group):>6} {100*c/len(group):>9.1f}%")

    # False positive analysis
    print(f"\n{'Action':<12} {'FP Count':>9}  (predicted this action when it was wrong)")
    print("-" * 50)
    fp_counts: dict[str, int] = defaultdict(int)
    for r in results:
        if not r.correct_action:
            fp_counts[r.predicted_action] += 1
    for action in sorted(fp_counts, key=fp_counts.get, reverse=True):
        print(f"{action:<12} {fp_counts[action]:>9}")

    # Detail table
    print(f"\n{'ID':<5} {'Cat':<13} {'Expected':<12} {'Predicted':<12} {'Conf':>5} {'ms':>6} {'OK':>3}")
    print("-" * 65)
    for i, r in enumerate(results):
        # Try to get test case ID from index
        mark = "Y" if r.correct_action else "N"
        cat = ""
        # Extract category from query prefix if available
        query_short = r.query[:30] + ("..." if len(r.query) > 30 else "")
        exp = r.expected_action
        if r.expected_target:
            exp += f":{r.expected_target}"
        pred = r.predicted_action
        if r.predicted_target:
            pred += f":{r.predicted_target}"
        print(f"{i+1:<5} {query_short:<30} {exp:<15} {pred:<15} {r.confidence:>4.2f} {r.latency_ms:>5.0f} {mark:>3}")

    # Failures detail
    failures = [r for r in results if not r.correct_action]
    if failures:
        print(f"\n--- FAILURES ({len(failures)}) ---")
        for r in failures:
            print(f"\n  Query: {r.query}")
            print(f"  Expected: {r.expected_action}, Got: {r.predicted_action} (conf={r.confidence:.3f})")
            print(f"  Top-k: {r.top_k_labels}")

    # Comparison baseline
    print("\n--- BASELINE COMPARISON ---")
    print(f"nvidia_orchestrator-8b: 10/10 (571ms avg, 578 tokens)")
    print(f"Concept router:        {correct_action}/{total} "
          f"({np.mean(latencies):.0f}ms avg, 0 tokens)")

    return results


def save_results(results: list[ClassificationResult], output_path: str):
    data = {
        "experiment": "L1",
        "description": "Concept-level router accuracy vs LLM router",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total": len(results),
            "correct_action": sum(1 for r in results if r.correct_action),
            "correct_both": sum(1 for r in results if r.correct_action and r.correct_target),
            "mean_latency_ms": float(np.mean([r.latency_ms for r in results])),
            "median_latency_ms": float(np.median([r.latency_ms for r in results])),
            "p95_latency_ms": float(np.percentile([r.latency_ms for r in results], 95)),
            "tokens_consumed": 0,
        },
        "baseline": {
            "model": "nvidia_orchestrator-8b",
            "accuracy": "10/10",
            "avg_latency_ms": 571,
            "tokens_per_call": 578,
        },
        "results": [
            {
                "query": r.query,
                "expected_action": r.expected_action,
                "expected_target": r.expected_target,
                "predicted_action": r.predicted_action,
                "predicted_target": r.predicted_target,
                "confidence": round(r.confidence, 4),
                "correct_action": r.correct_action,
                "correct_target": r.correct_target,
                "latency_ms": round(r.latency_ms, 2),
                "top_k": r.top_k_labels,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="L1 Concept Router Test Bench")
    parser.add_argument("--backend", choices=["local", "lmstudio"], default="local",
                        help="Embedding backend (default: local sentence-transformers)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--host", default="192.168.1.22:1234",
                        help="LM Studio host:port (default: 192.168.1.22:1234)")
    parser.add_argument("--strategy", choices=["centroid", "knn"], default="centroid",
                        help="Classification strategy (default: centroid)")
    parser.add_argument("--k", type=int, default=5,
                        help="k for k-NN (default: 5)")
    parser.add_argument("--reference", default=None,
                        help="Path to reference set JSON")
    parser.add_argument("--test", default=None,
                        help="Path to test set JSON")
    parser.add_argument("--output", default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    ref_path = args.reference or str(base_dir / "reference_set.json")
    test_path = args.test or str(base_dir / "test_set.json")
    output_path = args.output or str(base_dir / "results" / "L1_results.json")

    # Create backend
    if args.backend == "lmstudio":
        backend = LMStudioBackend(host=args.host, model=args.model if args.model != "all-MiniLM-L6-v2" else None)
    else:
        backend = SentenceTransformerBackend(model_name=args.model)

    # Build router
    router = ConceptRouter(backend, ref_path, k=args.k, strategy=args.strategy)

    # Run evaluation
    print(f"\nRunning {test_path} ...")
    results = evaluate(router, test_path)
    print_report(results)
    save_results(results, output_path)

    # Exit code based on success criteria (>=9/10 = 90% on 30 cases = >=27)
    accuracy = sum(1 for r in results if r.correct_action) / len(results)
    threshold = 0.9
    if accuracy >= threshold:
        print(f"\nSUCCESS: {accuracy:.1%} >= {threshold:.0%} threshold. Proceed to L2.")
        sys.exit(0)
    elif accuracy >= 0.7:
        print(f"\nPARTIAL: {accuracy:.1%} — analyze failures, add reference examples, try hybrid.")
        sys.exit(1)
    else:
        print(f"\nFAIL: {accuracy:.1%} < 70% — concept routing insufficient for this action space.")
        sys.exit(2)


if __name__ == "__main__":
    main()
