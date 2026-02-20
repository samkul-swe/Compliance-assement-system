"""
ThresholdLearner — Simplified Medium+ Logic

We only care about medium, high, and critical violations.
Low and compliant are treated the same — no action needed.

Compliance score = sim(sentence, violation_pool) - sim(sentence, compliant_pool)
    Positive = closer to violation space
    Negative = closer to compliant space

One gate:
    Below medium threshold → no action (compliant or low, we don't care)
    Above medium threshold → violation worth flagging
    Near medium threshold  → Layer 2 to confirm

For sentences above threshold:
    Compare to medium/high/critical means → closest one wins
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple, Optional
import json
from pathlib import Path

SEVERITY_ORDER    = ['compliant', 'low', 'medium', 'high', 'critical']
FLAGGED_SEVERITIES = ['medium', 'high', 'critical']  # only these get warnings


class ThresholdLearner:

    def __init__(self, model: SentenceTransformer, severity_examples: Dict[str, list]):
        self.model = model
        self.severity_examples = severity_examples
        self.violation_pool_mean = None
        self.compliant_pool_mean = None
        self.severity_means = {}   # severity → mean compliance score
        self.medium_threshold = None  # the one gate
        self._learn()

    def _compliance_score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        score = sim(embedding, violation_pool) - sim(embedding, compliant_pool)
        Positive = violation territory. Negative = compliant territory.
        """
        v_sims = cosine_similarity(
            embeddings,
            self.violation_pool_mean.reshape(1, -1)
        ).flatten()

        c_sims = cosine_similarity(
            embeddings,
            self.compliant_pool_mean.reshape(1, -1)
        ).flatten()

        return v_sims - c_sims

    def _learn(self):
        print("Learning severity anchors from training data...")

        # Build violation pool mean — all non-compliant examples
        violation_phrases = [
            p for sev, phrases in self.severity_examples.items()
            if sev != 'compliant'
            for p in phrases
        ]
        self.violation_pool_mean = np.mean(
            self.model.encode(violation_phrases, convert_to_numpy=True),
            axis=0
        )

        # Build compliant pool mean
        self.compliant_pool_mean = np.mean(
            self.model.encode(
                self.severity_examples['compliant'], convert_to_numpy=True
            ),
            axis=0
        )

        # Compute mean compliance score per severity
        print(f"\n  {'Severity':10}  {'Min':>7}  {'Max':>7}  {'Mean':>7}  {'Std':>6}")
        print(f"  {'-'*50}")

        for severity in SEVERITY_ORDER:
            if severity not in self.severity_examples:
                continue
            embs   = self.model.encode(
                self.severity_examples[severity], convert_to_numpy=True
            )
            scores = self._compliance_score(embs)
            self.severity_means[severity] = float(np.mean(scores))

            print(
                f"  {severity:10}  "
                f"{np.min(scores):>7.3f}  "
                f"{np.max(scores):>7.3f}  "
                f"{np.mean(scores):>7.3f}  "
                f"{np.std(scores):>6.3f}"
            )

        # The gate sits at the midpoint between low mean and medium mean
        # This is the natural boundary between what we flag and what we ignore
        low_mean    = self.severity_means.get('low', 0.0)
        medium_mean = self.severity_means.get('medium', 0.0)
        self.medium_threshold = (low_mean + medium_mean) / 2

        print(f"\n  Low mean:        {low_mean:.3f}")
        print(f"  Medium mean:     {medium_mean:.3f}")
        print(f"  Gate (midpoint): {self.medium_threshold:.3f}")
        print(f"  Below gate → no action")
        print(f"  Above gate → violation → Layer 2 determines severity")

        # Verify ordering makes sense
        print(f"\n  Score ordering (should increase low → critical):")
        for sev in SEVERITY_ORDER:
            if sev in self.severity_means:
                bar = "█" * max(0, int((self.severity_means[sev] + 0.3) * 40))
                print(f"  {sev:10}  {self.severity_means[sev]:>7.3f}  {bar}")

        print("\n  Anchors learned ✓\n")

    def classify(
        self,
        embedding: np.ndarray,
        boundary_margin: float = 0.02
    ) -> Tuple[str, float, Optional[str]]:
        """
        Binary classification only.

        Below gate          → no_action
        Near gate (margin)  → violation, sitting_between set → Layer 2
        Above gate          → violation, Layer 2 determines severity
        """
        score = float(self._compliance_score(embedding.reshape(1, -1))[0])

        if score < self.medium_threshold - boundary_margin:
            return 'no_action', score, None

        if score < self.medium_threshold + boundary_margin:
            # Near the gate — uncertain, Layer 2 decides
            return 'violation', score, 'no_action and medium'

        # Clearly above gate — violation confirmed
        # Severity to be determined by Layer 2
        return 'violation', score, None

    def save_boundaries(self, path: str = "data/learned_ranges.json"):
        Path("data").mkdir(exist_ok=True)
        data = {
            'medium_threshold': self.medium_threshold,
            'severity_means':   self.severity_means
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Ranges saved to {path}")

    def save_ranges(self, path: str = "data/learned_ranges.json"):
        self.save_boundaries(path)