"""
Fine Tuning Script for Compliance SBERT

Loads training pairs from data/training_pairs.json
Fine tunes all-MiniLM-L6-v2 on your compliance domain
Saves best model to models/compliance_sbert

Run order:
    1. pair_generator.py   → generates data/training_pairs.json
    2. fine_tune.py        → trains and saves models/compliance_sbert
    3. layer1_checker.py   → uses fine tuned model automatically
"""

import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CONFIG = {
    "base_model":       "all-MiniLM-L6-v2",   # same model Layer 1 uses
    "output_path":      "models/compliance_sbert",
    "training_pairs":   "data/training_pairs.json",
    "epochs":           10,
    "batch_size":       16,
    "warmup_steps":     50,
    "test_size":        0.2,                   # 20% held out for evaluation
    "random_seed":      42,
}

# These are the sanity check cases run before and after training
# to visually confirm the model improved
SANITY_CHECKS = [
    {
        "description": "Same violation, different phrasing → should be HIGH",
        "text_a": "We're going to sue you if you don't pay",
        "text_b": "Our legal department will be in touch shortly",
        "want": "HIGH"
    },
    {
        "description": "Critical violation vs compliant → should be LOW",
        "text_a": "We're going to sue you if you don't pay",
        "text_b": "Let me look into a payment plan for you",
        "want": "LOW"
    },
    {
        "description": "Third party threat, paraphrased → should be HIGH",
        "text_a": "Your employer will be notified about this debt",
        "text_b": "We'll be speaking to your workplace about this matter",
        "want": "HIGH"
    },
    {
        "description": "Low violation vs compliant → should be LOW",
        "text_a": "You really should take care of this as soon as possible",
        "text_b": "Take all the time you need, there's no rush",
        "want": "LOW"
    },
    {
        "description": "Critical vs low violation → should be MEDIUM (neighbouring severity)",
        "text_a": "We will take you to court if this isn't paid today",
        "text_b": "I really do need you to commit to something today",
        "want": "MEDIUM"
    },
    {
        "description": "Empathy vs dismissiveness → should be LOW",
        "text_a": "I completely understand this is a really difficult situation",
        "text_b": "That's not our problem, figure it out yourself",
        "want": "LOW"
    },
]


# ─────────────────────────────────────────────
# Load pairs
# ─────────────────────────────────────────────

def load_pairs(path: str) -> List[InputExample]:
    """Load training pairs from JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Training pairs not found at {path}\n"
            f"Run pair_generator.py first."
        )

    with open(p, 'r') as f:
        data = json.load(f)

    pairs = [
        InputExample(texts=[d['text_a'], d['text_b']], label=float(d['label']))
        for d in data
    ]

    # Print label distribution so you can see the balance
    label_counts = {}
    for p_ in pairs:
        l = str(round(p_.label, 2))
        label_counts[l] = label_counts.get(l, 0) + 1

    logger.info(f"Loaded {len(pairs)} training pairs:")
    for label, count in sorted(label_counts.items()):
        bar = "█" * (count // 20)
        logger.info(f"  label={label}: {count:4d} pairs  {bar}")

    return pairs


# ─────────────────────────────────────────────
# Sanity check — run before and after training
# ─────────────────────────────────────────────

def run_sanity_checks(model: SentenceTransformer, label: str = ""):
    """
    Run known test cases and print similarity scores.
    Call this before and after training to see the difference.
    """
    print(f"\n{'─'*70}")
    print(f"SANITY CHECKS {label}")
    print(f"{'─'*70}")

    results = []
    for case in SANITY_CHECKS:
        embs = model.encode([case['text_a'], case['text_b']])
        sim = float(cosine_similarity([embs[0]], [embs[1]])[0][0])

        # Interpret score
        if sim >= 0.70:
            actual = "HIGH"
        elif sim >= 0.40:
            actual = "MEDIUM"
        else:
            actual = "LOW"

        correct = actual == case['want']
        icon = "✅" if correct else "❌"

        print(f"\n{icon} {case['description']}")
        print(f"   A: {case['text_a'][:65]}")
        print(f"   B: {case['text_b'][:65]}")
        print(f"   Similarity: {sim:.3f}  |  Expected: {case['want']}  |  Got: {actual}")

        results.append(correct)

    correct_count = sum(results)
    print(f"\n{'─'*70}")
    print(f"Passed: {correct_count}/{len(results)}")
    print(f"{'─'*70}\n")

    return correct_count


# ─────────────────────────────────────────────
# Fine tune
# ─────────────────────────────────────────────

def fine_tune(pairs: List[InputExample]) -> SentenceTransformer:
    """
    Fine tune the base model on compliance pairs.

    Steps:
    1. Split pairs into train and test (test never seen during training)
    2. Load base model
    3. Run sanity checks on base model (before)
    4. Train using CosineSimilarityLoss + gradient descent
    5. Evaluate on test set after each epoch — saves best epoch
    6. Run sanity checks on fine tuned model (after)
    7. Print comparison
    """
    cfg = CONFIG
    random.seed(cfg['random_seed'])

    # Split — 80% train, 20% test
    # Fixed seed means split is reproducible across runs
    train_pairs, test_pairs = train_test_split(
        pairs,
        test_size=cfg['test_size'],
        random_state=cfg['random_seed']
    )

    logger.info(f"Train: {len(train_pairs)} pairs  |  Test: {len(test_pairs)} pairs")

    # Load base model
    logger.info(f"Loading base model: {cfg['base_model']}...")
    model = SentenceTransformer(cfg['base_model'])

    # Sanity check BEFORE training — baseline scores
    before_score = run_sanity_checks(model, label="(BEFORE FINE TUNING)")

    # DataLoader — feeds pairs in shuffled batches of 16
    # shuffle=True means each epoch sees pairs in a different order
    train_dataloader = DataLoader(
        train_pairs,
        shuffle=True,
        batch_size=cfg['batch_size']
    )

    # Loss function
    # CosineSimilarityLoss: for each pair compute cosine similarity,
    # compare to label, loss = (cosine - label)²
    # Gradient descent nudges weights to reduce this loss
    train_loss = losses.CosineSimilarityLoss(model)

    # Evaluator — runs on test set after each epoch
    # Measures Pearson and Spearman correlation between
    # predicted similarity and true labels
    # save_best_model=True saves the epoch with best Spearman correlation
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=[p.texts[0] for p in test_pairs],
        sentences2=[p.texts[1] for p in test_pairs],
        scores=[p.label for p in test_pairs],
        name='compliance-test'
    )

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * cfg['epochs']

    logger.info(f"\nTraining configuration:")
    logger.info(f"  Epochs:        {cfg['epochs']}")
    logger.info(f"  Batch size:    {cfg['batch_size']}")
    logger.info(f"  Steps/epoch:   {steps_per_epoch}")
    logger.info(f"  Total steps:   {total_steps}")
    logger.info(f"  Warmup steps:  {cfg['warmup_steps']}")
    logger.info(f"  Output:        {cfg['output_path']}")
    logger.info(f"\nStarting fine tuning...")

    Path(cfg['output_path']).mkdir(parents=True, exist_ok=True)

    # This is gradient descent running on your compliance pairs
    # warmup_steps: learning rate ramps up for first 50 steps
    #               prevents overwriting general knowledge too aggressively
    # save_best_model: saves the epoch with best test set performance
    #                  protects against overfitting
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=cfg['epochs'],
        warmup_steps=cfg['warmup_steps'],
        output_path=cfg['output_path'],
        save_best_model=True,
        show_progress_bar=True
    )

    logger.info(f"\n✅ Fine tuning complete. Best model saved to {cfg['output_path']}")

    # Load the best saved model (might not be the last epoch)
    best_model = SentenceTransformer(cfg['output_path'])

    # Sanity check AFTER training — compare to baseline
    after_score = run_sanity_checks(best_model, label="(AFTER FINE TUNING)")

    # Final comparison
    print(f"\n{'='*70}")
    print("RESULT COMPARISON")
    print(f"{'='*70}")
    print(f"  Before fine tuning: {before_score}/{len(SANITY_CHECKS)} sanity checks passed")
    print(f"  After fine tuning:  {after_score}/{len(SANITY_CHECKS)} sanity checks passed")

    if after_score > before_score:
        print(f"\n  ✅ Fine tuning improved the model")
        print(f"  Run layer1_checker.py to measure full accuracy improvement")
    elif after_score == before_score:
        print(f"\n  ⚠️  No change on sanity checks")
        print(f"  Check your training pairs — may need more examples or epochs")
    else:
        print(f"\n  ❌ Model got worse on sanity checks")
        print(f"  Likely causes: too few pairs, imbalanced labels, too many epochs")

    print(f"{'='*70}\n")

    return best_model


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*70)
    print("COMPLIANCE SBERT FINE TUNER")
    print("="*70 + "\n")

    # Load pairs generated by pair_generator.py
    pairs = load_pairs(CONFIG['training_pairs'])

    if len(pairs) < 50:
        print("⚠️  Very few training pairs. Fine tuning may not generalise well.")
        print("   Add more phrases in pair_generator.py and regenerate.")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            exit()

    # Fine tune
    fine_tune(pairs)

    print("Next step: run layer1_checker.py")
    print("The checker will automatically load models/compliance_sbert")