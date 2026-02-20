"""
ThresholdLearner

Learns severity boundaries from labelled training examples.
Instead of hardcoded thresholds, computes min/max cosine similarity
for each severity level against its own centroid.

Result:
    Each severity level gets a [min, max] range.
    A new sentence is classified by which range its score falls into.
    If it sits on a boundary between two ranges → Layer 2.

Example output:
    compliant: [0.61, 0.89]
    low:       [0.42, 0.63]
    medium:    [0.55, 0.78]
    high:      [0.68, 0.91]
    critical:  [0.74, 0.97]
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple, Optional
import json
from pathlib import Path


SEVERITY_ORDER = ['compliant', 'low', 'medium', 'high', 'critical']


class ThresholdLearner:
    """
    Learns min/max cosine similarity boundaries per severity level
    from labelled training examples.

    At startup:
        1. Embed all training phrases per severity
        2. Compute centroid per severity
        3. Compute cosine similarity of each phrase against its own centroid
        4. Record min and max — these become the boundaries

    At inference:
        1. Embed new sentence
        2. Compute cosine similarity against all centroids
        3. Check which severity range the score falls into
        4. If near a boundary → escalate to Layer 2
    """

    def __init__(self, model: SentenceTransformer, severity_examples: Dict[str, list]):
        self.model = model
        self.severity_examples = severity_examples
        self.centroids = {}
        self.boundaries = {}   # severity → {'min': float, 'max': float}
        self._learn()

    def _learn(self):
        """
        Step 1: Compute centroids
        Step 2: For each severity, compute cosine similarity of every
                phrase against its own centroid
        Step 3: Record min and max as the boundary for that severity
        """
        print("Learning severity boundaries from training data...")

        # Step 1 — Embed all phrases and compute centroids
        all_embeddings = {}
        for severity, phrases in self.severity_examples.items():
            embeddings = self.model.encode(phrases, convert_to_numpy=True)
            all_embeddings[severity] = embeddings
            self.centroids[severity] = np.mean(embeddings, axis=0)

        # Step 2 & 3 — Compute similarities and learn boundaries
        print(f"\n  {'Severity':10}  {'Min':>6}  {'Max':>6}  {'Mean':>6}  {'Phrases':>7}")
        print(f"  {'-'*45}")

        for severity in SEVERITY_ORDER:
            if severity not in all_embeddings:
                continue

            embeddings = all_embeddings[severity]
            centroid   = self.centroids[severity]

            # Cosine similarity of every phrase against its own centroid
            sims = cosine_similarity(
                embeddings,
                centroid.reshape(1, -1)
            ).flatten()

            self.boundaries[severity] = {
                'min':  float(np.min(sims)),
                'max':  float(np.max(sims)),
                'mean': float(np.mean(sims)),
                'std':  float(np.std(sims))
            }

            print(
                f"  {severity:10}  "
                f"{self.boundaries[severity]['min']:>6.3f}  "
                f"{self.boundaries[severity]['max']:>6.3f}  "
                f"{self.boundaries[severity]['mean']:>6.3f}  "
                f"{len(embeddings):>7}"
            )

        print("\n  Boundaries learned ✓\n")

    def classify(
        self,
        embedding: np.ndarray,
        boundary_margin: float = 0.04
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Classify a sentence embedding against learned boundaries.

        For each severity centroid, compute cosine similarity.
        Check which severity's [min, max] range the score falls into.

        Returns:
            detected_severity: best matching severity or None
            best_score:        cosine similarity to best centroid
            sitting_between:   e.g. "low and medium" if near boundary, else None

        Logic:
            score >= min AND score <= max → in this severity's range
            score near boundary (within margin) → uncertain → Layer 2
            score below all minimums → compliant
        """
        # Compute similarity against every centroid
        scores = {}
        for severity, centroid in self.centroids.items():
            sim = cosine_similarity(
                embedding.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0][0]
            scores[severity] = float(sim)

        # Find which severity range each score falls into
        matches = []
        for severity in SEVERITY_ORDER:
            if severity not in self.boundaries:
                continue
            b = self.boundaries[severity]
            score = scores[severity]
            if b['min'] <= score <= b['max']:
                matches.append((severity, score))

        # No match → compliant (score falls below all known ranges)
        if not matches:
            return 'compliant', scores.get('compliant', 0.0), None

        # Single clear match
        if len(matches) == 1:
            severity, score = matches[0]

            # Check if near the boundary of adjacent severity
            sitting_between = self._check_boundary(
                severity, score, boundary_margin
            )
            return severity, score, sitting_between

        # Multiple matches → overlapping ranges → pick highest severity
        # but flag as sitting between
        matches.sort(key=lambda x: SEVERITY_ORDER.index(x[0]))
        lower = matches[0][0]
        upper = matches[-1][0]
        best_severity = upper
        best_score = scores[best_severity]

        sitting_between = f"{lower} and {upper}"
        return best_severity, best_score, sitting_between

    def _check_boundary(
        self,
        severity: str,
        score: float,
        margin: float
    ) -> Optional[str]:
        """
        Check if a score sits near the boundary between two
        adjacent severity levels.

        Returns "lower and upper" string if near boundary, else None.
        """
        idx = SEVERITY_ORDER.index(severity)
        b = self.boundaries[severity]

        # Check upper boundary — near max of this severity?
        if idx < len(SEVERITY_ORDER) - 1:
            upper_severity = SEVERITY_ORDER[idx + 1]
            if upper_severity in self.boundaries:
                upper_b = self.boundaries[upper_severity]
                # Near top of current range or bottom of next range
                if (b['max'] - score) < margin or (score - upper_b['min']) < margin:
                    return f"{severity} and {upper_severity}"

        # Check lower boundary — near min of this severity?
        if idx > 0:
            lower_severity = SEVERITY_ORDER[idx - 1]
            if lower_severity in self.boundaries:
                # Near bottom of current range
                if (score - b['min']) < margin:
                    return f"{lower_severity} and {severity}"

        return None

    def save_boundaries(self, path: str = "data/learned_boundaries.json"):
        """Save learned boundaries for inspection."""
        Path("data").mkdir(exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.boundaries, f, indent=2)
        print(f"Boundaries saved to {path}")

    def print_boundaries(self):
        """Print a summary of learned boundaries."""
        print("\nLearned Severity Boundaries:")
        print(f"{'Severity':10}  {'Range':>20}")
        print("-" * 35)
        for sev in SEVERITY_ORDER:
            if sev in self.boundaries:
                b = self.boundaries[sev]
                print(f"{sev:10}  [{b['min']:.3f}  →  {b['max']:.3f}]")