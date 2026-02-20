"""
ThresholdLearner — Position Based

SBERT already places sentences on a compliance spectrum.
Compliant language sits near 0, violation language near 1.

At training time:
    For each known phrase, compute its cosine similarity against
    the combined violation pool. This gives every phrase a single
    score representing its position on the compliance spectrum.
    Record min and max per severity → those are the severity ranges.

At inference time:
    Compute new sentence's similarity against the same violation pool.
    Check which severity range that score falls into.
    If in overlap between two adjacent ranges → Layer 2.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple, Optional, List
import json
from pathlib import Path

SEVERITY_ORDER = ['compliant', 'low', 'medium', 'high', 'critical']


class ThresholdLearner:

    def __init__(self, model: SentenceTransformer, severity_examples: Dict[str, list]):
        self.model = model
        self.severity_examples = severity_examples
        self.violation_pool = None      # combined violation embeddings
        self.violation_pool_mean = None # mean vector of violation pool
        self.ranges = {}                # severity → {'min': float, 'max': float}
        self._learn()

    def _learn(self):
        """
        Step 1 — Build the violation pool
            All non-compliant examples combined into one pool.
            This represents the violation end of the spectrum.

        Step 2 — Score every known phrase
            Each phrase gets one score: cosine similarity against
            the mean of the violation pool.
            This places every phrase on the 0→1 compliance spectrum.

        Step 3 — Record min/max per severity
            These are the learned ranges.
        """
        print("Learning severity ranges from training data...")

        # Step 1 — Build violation pool from all non-compliant examples
        violation_phrases = [
            p for sev, phrases in self.severity_examples.items()
            if sev != 'compliant'
            for p in phrases
        ]
        violation_embs = self.model.encode(
            violation_phrases, convert_to_numpy=True
        )
        self.violation_pool_mean = np.mean(violation_embs, axis=0)

        # Step 2 & 3 — Score each severity's phrases and record ranges
        print(f"\n  {'Severity':10}  {'Min':>6}  {'Max':>6}  {'Mean':>6}  {'Phrases':>7}")
        print(f"  {'-'*48}")

        all_scores = {}

        for severity in SEVERITY_ORDER:
            if severity not in self.severity_examples:
                continue

            phrases = self.severity_examples[severity]
            embs    = self.model.encode(phrases, convert_to_numpy=True)

            # Score each phrase against violation pool mean
            # Higher score = closer to violation end of spectrum
            scores = cosine_similarity(
                embs,
                self.violation_pool_mean.reshape(1, -1)
            ).flatten()

            all_scores[severity] = scores

            self.ranges[severity] = {
                'min':  float(np.min(scores)),
                'max':  float(np.max(scores)),
                'mean': float(np.mean(scores)),
                'std':  float(np.std(scores))
            }

            print(
                f"  {severity:10}  "
                f"{self.ranges[severity]['min']:>6.3f}  "
                f"{self.ranges[severity]['max']:>6.3f}  "
                f"{self.ranges[severity]['mean']:>6.3f}  "
                f"{len(phrases):>7}"
            )

        print("\n  Ranges learned ✓\n")

        # Print overlap regions — these are the grey zones for Layer 2
        print("  Grey zones (overlapping ranges):")
        for i in range(len(SEVERITY_ORDER) - 1):
            lower = SEVERITY_ORDER[i]
            upper = SEVERITY_ORDER[i + 1]
            if lower not in self.ranges or upper not in self.ranges:
                continue
            overlap_start = self.ranges[upper]['min']
            overlap_end   = self.ranges[lower]['max']
            if overlap_start < overlap_end:
                print(f"  {lower} / {upper}: [{overlap_start:.3f} → {overlap_end:.3f}]")
        print()

    def score(self, embedding: np.ndarray) -> float:
        """
        Get a single compliance score for a sentence embedding.
        Higher = closer to violation space.
        """
        return float(cosine_similarity(
            embedding.reshape(1, -1),
            self.violation_pool_mean.reshape(1, -1)
        )[0][0])

    def classify(
        self,
        embedding: np.ndarray,
        boundary_margin: float = 0.02
    ) -> Tuple[str, float, Optional[str]]:
        """
        Classify a sentence using its position on the compliance spectrum.

        1. Compute score against violation pool mean
        2. Check which severity range it falls into
        3. If in overlap between adjacent ranges → sitting_between

        Returns:
            detected_severity: matched severity level
            score:             position on compliance spectrum
            sitting_between:   e.g. "low and medium" if in grey zone
        """
        score = self.score(embedding)

        # Find which ranges this score falls into
        matches = [
            sev for sev in SEVERITY_ORDER
            if sev in self.ranges
            and self.ranges[sev]['min'] <= score <= self.ranges[sev]['max']
        ]

        # No match at all — below all ranges → compliant
        if not matches:
            # If score is below compliant min → definitely compliant
            # If score is above critical max → definitely critical
            if score <= self.ranges.get('compliant', {}).get('min', 0):
                return 'compliant', score, None
            if score >= self.ranges.get('critical', {}).get('max', 1):
                return 'critical', score, None
            # Between ranges but no match → return closest
            return self._closest_range(score), score, None

        # Single clear match
        if len(matches) == 1:
            severity = matches[0]
            # Check if near boundary of adjacent severity
            sitting_between = self._check_near_boundary(
                score, severity, boundary_margin
            )
            return severity, score, sitting_between

        # Multiple matches → score in overlap → sitting between
        # Sort by severity order and return the overlap description
        matches_sorted = sorted(matches, key=lambda s: SEVERITY_ORDER.index(s))
        lower = matches_sorted[0]
        upper = matches_sorted[-1]

        # Only flag as sitting_between if adjacent
        if SEVERITY_ORDER.index(upper) - SEVERITY_ORDER.index(lower) == 1:
            return upper, score, f"{lower} and {upper}"

        # Non-adjacent overlap — pick highest severity
        return upper, score, None

    def _closest_range(self, score: float) -> str:
        """Find closest severity range when score falls between ranges."""
        min_dist = float('inf')
        closest  = 'compliant'
        for sev, r in self.ranges.items():
            # Distance to nearest edge of range
            dist = min(abs(score - r['min']), abs(score - r['max']))
            if dist < min_dist:
                min_dist = dist
                closest  = sev
        return closest

    def _check_near_boundary(
        self,
        score: float,
        severity: str,
        margin: float
    ) -> Optional[str]:
        """
        Check if score sits within margin of the boundary
        between this severity and an adjacent one.
        """
        idx = SEVERITY_ORDER.index(severity)
        r   = self.ranges[severity]

        # Check upper boundary
        if idx < len(SEVERITY_ORDER) - 1:
            upper = SEVERITY_ORDER[idx + 1]
            if upper in self.ranges:
                upper_min = self.ranges[upper]['min']
                # Near top of this range or bottom of next
                if abs(score - upper_min) < margin or (r['max'] - score) < margin:
                    return f"{severity} and {upper}"

        # Check lower boundary
        if idx > 0:
            lower = SEVERITY_ORDER[idx - 1]
            if lower in self.ranges:
                lower_max = self.ranges[lower]['max']
                # Near bottom of this range or top of previous
                if abs(score - lower_max) < margin or (score - r['min']) < margin:
                    return f"{lower} and {severity}"

        return None

    def save_ranges(self, path: str = "data/learned_ranges.json"):
        Path("data").mkdir(exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.ranges, f, indent=2)
        print(f"Ranges saved to {path}")

    def save_boundaries(self, path: str = "data/learned_ranges.json"):
        """Alias for save_ranges — keeps compatibility."""
        self.save_ranges(path)

    def print_ranges(self):
        print("\nLearned Severity Ranges (on compliance spectrum):")
        print(f"{'Severity':10}  {'Range':>25}")
        print("-" * 40)
        for sev in SEVERITY_ORDER:
            if sev in self.ranges:
                r = self.ranges[sev]
                print(f"{sev:10}  [{r['min']:.3f}  →  {r['max']:.3f}]")