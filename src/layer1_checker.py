"""
Layer 1: Semantic Compliance Checker (Fine Tuned)

Changes from original:
- Uses fine tuned compliance SBERT model
- Removed flat semantic threshold (0.60 gate)
- Removed keyword matching entirely
- Confidence derived from boundary distance + score separation
- Severity thresholds now directly on cosine similarity (0.0-1.0 scale)
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ComplianceViolation:
    rule_id: str
    category: str
    severity: str
    description: str
    message_index: int
    matched_text: str
    similarity_score: float


@dataclass
class ComplianceResult:
    conversation_id: str
    compliant: bool
    confidence: float
    threshold: float
    needs_llm_review: bool
    violations: List[ComplianceViolation]
    evidence: List[str]
    similarity_scores: Dict[str, float]
    agent_message_count: int
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            'conversation_id': self.conversation_id,
            'compliant': bool(self.compliant),
            'confidence': float(self.confidence),
            'threshold': float(self.threshold),
            'needs_llm_review': bool(self.needs_llm_review),
            'violations': [
                {
                    'rule_id': v.rule_id,
                    'category': v.category,
                    'severity': v.severity,
                    'description': v.description,
                    'message_index': v.message_index,
                    'matched_text': v.matched_text,
                    'similarity_score': float(v.similarity_score),
                }
                for v in self.violations
            ],
            'evidence': self.evidence,
            'similarity_scores': {k: float(v) for k, v in self.similarity_scores.items()},
            'agent_message_count': self.agent_message_count,
            'timestamp': self.timestamp
        }


class ConfigLoader:

    @staticmethod
    def load_compliance_rules(rules_file: str = "data/compliance_rules.json") -> Dict:
        path = Path(rules_file)
        if not path.exists():
            raise FileNotFoundError(f"Compliance rules not found: {rules_file}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_confidence_config(config_file: str = "config/severity_confidence.json") -> Dict:
        path = Path(config_file)
        if not path.exists():
            logger.warning(f"Config not found: {config_file}, using defaults")
            return ConfigLoader._default_config()
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _default_config() -> Dict:
        return {
            "severity_thresholds": {
                "compliant": 0.20,
                "low":       0.35,
                "medium":    0.55,
                "high":      0.70,
                "critical":  0.82
            },
            "escalation_confidence_cutoff": 0.70
        }


class SemanticComplianceChecker:
    """
    Layer 1: Semantic compliance checker using fine tuned compliance SBERT.

    How it works:
    1. Load fine tuned model (trained on compliance severity spectrum 0.0-1.0)
    2. Embed all violation examples per rule (done once at startup)
    3. For each agent message, compute cosine similarity against all rule examples
    4. Take max similarity per rule - this score lives on your severity spectrum
    5. Compare directly against severity threshold - no flat gate, no keywords
    6. Confidence = how far the score sits from the threshold boundary
    7. Low confidence → escalate to Layer 2
    """

    def __init__(self,
                 rules_file: str = "data/compliance_rules.json",
                 config_file: str = "config/severity_confidence.json",
                 model_path: str = "models/compliance_sbert"):
        logger.info("Initializing Semantic Compliance Checker (Fine Tuned)...")

        self.rules_data = ConfigLoader.load_compliance_rules(rules_file)
        self.config = ConfigLoader.load_confidence_config(config_file)
        self.rules = {rule['id']: rule for rule in self.rules_data['rules']}

        logger.info(f"Loaded {len(self.rules)} compliance rules")

        # Use fine tuned model if available, fall back to base
        if Path(model_path).exists():
            logger.info(f"Loading fine tuned model from {model_path}...")
            self.embedding_model = SentenceTransformer(model_path)
            logger.info("Fine tuned model loaded ✓")
        else:
            logger.warning(f"Fine tuned model not found at {model_path}")
            logger.warning("Falling back to base model — run fine_tune.py first")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self._build_examples()
        logger.info("✅ Checker initialized\n")

    def _build_examples(self):
        """Build and embed violation examples per rule."""
        self.violation_examples = {
            'R001': [
                "We're going to sue you if you don't pay",
                "I'll have your wages garnished",
                "We're sending this to our lawyers",
                "You'll be taken to court over this",
                "Legal action will be taken against you",
                "We can have you arrested for non-payment",
                "Our legal department will be in touch shortly",
                "This will be passed to our litigation team",
                "You'll be hearing from our solicitors",
                "A county court judgment will be filed against you"
            ],
            'R002': [
                "We'll contact your family about this debt",
                "Your employer will be notified",
                "We're going to call your boss",
                "Your family will find out about this",
                "We'll tell everyone you know",
                "Your workplace will receive a letter from us",
                "We'll be speaking to your next of kin",
                "Everyone on your application will be contacted",
                "Your spouse will be informed about this debt",
                "We have your employer's details and will be in touch"
            ],
            'R003': [
                "That's not our problem, figure it out yourself",
                "This is your fault for not paying",
                "I don't care about your excuses",
                "That's not my problem, deal with it",
                "You have to find a way, that's on you",
                "Stop wasting my time with these stories",
                "Everyone has problems, you still owe the money",
                "You're being deliberately difficult about this",
                "I find your attitude quite unacceptable",
                "Other customers manage to pay, why can't you"
            ],
            'R004': [
                "You need to pay immediately",
                "Pay right now or face consequences",
                "We need payment in full by tomorrow",
                "You must pay this today",
                "Pay immediately, no exceptions",
                "The full balance must be cleared today",
                "Payment in full is required before this call ends",
                "We cannot accept anything less than full payment now",
                "You have to clear this immediately, no options",
                "We require same-day payment of the full amount"
            ],
            'R005': [
                "Your credit will be ruined forever",
                "You'll never get a loan again",
                "We're going to destroy your credit rating",
                "This will ruin your financial future",
                "You'll never be able to buy a house",
                "No bank will ever lend to you after this",
                "Your credit file will be permanently damaged",
                "You'll be blacklisted from every financial institution",
                "This debt means you'll never get approved for anything",
                "Your financial life is essentially over if this isn't paid"
            ],
            'R006': [
                "That's not relevant to what we're discussing",
                "Your personal circumstances don't change what you owe",
                "We can't consider personal situations, the debt is the debt",
                "That's unfortunate but it doesn't affect the balance",
                "Personal issues aren't something we can factor in",
                "We deal with the account, not personal circumstances",
                "Whatever is happening in your life, the debt still stands",
                "That doesn't affect our position on this account",
                "I can't make decisions based on personal circumstances",
                "Regardless of your situation, the amount remains outstanding"
            ]
        }

        logger.info("Computing embeddings for violation examples...")
        self.violation_embeddings = {}
        for rule_id, examples in self.violation_examples.items():
            self.violation_embeddings[rule_id] = self.embedding_model.encode(
                examples, convert_to_numpy=True
            )
        logger.info(f"Embedded {sum(len(e) for e in self.violation_examples.values())} examples")

    def check_conversation(self, conversation: Dict) -> ComplianceResult:
        conv_id = conversation['conversation_id']

        agent_messages = [
            (i, msg['text']) for i, msg in enumerate(conversation['messages'])
            if msg['role'] == 'agent'
        ]

        if not agent_messages:
            return ComplianceResult(
                conversation_id=conv_id,
                compliant=True,
                confidence=1.0,
                threshold=self.config['severity_thresholds']['compliant'],
                needs_llm_review=False,
                violations=[],
                evidence=[],
                similarity_scores={},
                agent_message_count=0,
                timestamp=datetime.now().isoformat()
            )

        agent_texts = [text for _, text in agent_messages]
        agent_embeddings = self.embedding_model.encode(agent_texts, convert_to_numpy=True)

        violations = []
        evidence_list = []
        similarity_scores = {}

        for rule_id, rule_embeddings in self.violation_embeddings.items():
            if rule_id not in self.rules:
                continue

            max_similarity = 0.0
            best_match_idx = None
            best_match_text = None

            for (orig_idx, msg_text), agent_emb in zip(agent_messages, agent_embeddings):
                sims = cosine_similarity(agent_emb.reshape(1, -1), rule_embeddings)[0]
                msg_max = sims.max()
                if msg_max > max_similarity:
                    max_similarity = msg_max
                    best_match_idx = orig_idx
                    best_match_text = msg_text

            similarity_scores[rule_id] = round(float(max_similarity), 3)

            rule = self.rules[rule_id]
            severity = rule['severity']
            threshold = self.config['severity_thresholds'].get(
                severity,
                self.config['severity_thresholds']['medium']
            )

            # Direct comparison — no flat gate, no keyword confirmation
            if max_similarity >= threshold:
                violations.append(ComplianceViolation(
                    rule_id=rule_id,
                    category=rule['category'],
                    severity=severity,
                    description=rule['description'],
                    message_index=best_match_idx,
                    matched_text=best_match_text[:100] if best_match_text else "",
                    similarity_score=max_similarity,
                ))

                evidence_list.append(
                    f"{rule_id} ({severity}): \"{best_match_text[:80]}...\" "
                    f"(similarity: {max_similarity:.3f}, threshold: {threshold:.2f})"
                )

        confidence = self._calculate_confidence(violations, similarity_scores)
        compliant = len(violations) == 0
        threshold = self._get_threshold(violations)
        needs_review = confidence < self.config['escalation_confidence_cutoff']

        return ComplianceResult(
            conversation_id=conv_id,
            compliant=compliant,
            confidence=confidence,
            threshold=threshold,
            needs_llm_review=needs_review,
            violations=violations,
            evidence=evidence_list,
            similarity_scores=similarity_scores,
            agent_message_count=len(agent_messages),
            timestamp=datetime.now().isoformat()
        )

    def _calculate_confidence(
        self,
        violations: List[ComplianceViolation],
        similarity_scores: Dict[str, float]
    ) -> float:
        """
        Confidence derived from two factors:

        1. Boundary distance (60% weight)
           How far is the score from its severity threshold?
           Score 0.95 vs critical threshold 0.82 → distance 0.13 → confident
           Score 0.83 vs critical threshold 0.82 → distance 0.01 → uncertain

        2. Separation (40% weight)
           How much higher is the triggered rule vs everything else?
           One rule at 0.91, others at 0.15 → clear signal → confident
           Multiple rules clustering near 0.75 → ambiguous → uncertain
        """
        scores = list(similarity_scores.values())

        if not violations:
            max_score = max(scores) if scores else 0.0
            highest_rule = max(similarity_scores, key=similarity_scores.get)
            severity = self.rules.get(highest_rule, {}).get('severity', 'medium')
            threshold = self.config['severity_thresholds'].get(severity, 0.55)
            distance = threshold - max_score
            confidence = min(distance / threshold, 1.0)

        else:
            max_triggered = max(v.similarity_score for v in violations)
            threshold = self._get_threshold(violations)

            # Factor 1: boundary distance
            boundary_distance = max_triggered - threshold
            boundary_confidence = min(boundary_distance / (1.0 - threshold + 1e-6), 1.0)

            # Factor 2: separation from non-triggered rules
            triggered_ids = {v.rule_id for v in violations}
            non_triggered = [s for rid, s in similarity_scores.items() if rid not in triggered_ids]
            avg_other = np.mean(non_triggered) if non_triggered else 0.0
            separation = max_triggered - avg_other
            separation_confidence = min(separation / 0.5, 1.0)

            confidence = (boundary_confidence * 0.6) + (separation_confidence * 0.4)

        return round(float(np.clip(confidence, 0.0, 1.0)), 3)

    def _get_threshold(self, violations: List[ComplianceViolation]) -> float:
        severity_order = ['low', 'medium', 'high', 'critical']
        thresholds = self.config['severity_thresholds']
        if not violations:
            return thresholds['compliant']
        highest = max(
            (v.severity for v in violations),
            key=lambda s: severity_order.index(s) if s in severity_order else -1
        )
        return thresholds.get(highest, thresholds['medium'])

    def check_multiple(self, conversations: List[Dict]) -> List[ComplianceResult]:
        results = []
        for i, conv in enumerate(conversations, 1):
            if i % 20 == 0:
                logger.info(f"Processed {i}/{len(conversations)} conversations...")
            try:
                results.append(self.check_conversation(conv))
            except Exception as e:
                logger.error(f"Error checking {conv.get('conversation_id')}: {e}")
        return results


class OutputGenerator:

    def __init__(self, output_dir: str = "data/layer1_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_all(self, results: List[ComplianceResult]):
        logger.info("\nGenerating outputs...")
        auto = [r for r in results if not r.needs_llm_review]
        review = [r for r in results if r.needs_llm_review]
        self._save_json(auto, "auto_decided.json")
        self._save_json(review, "llm_review_queue.json")
        self._save_excel(results, "compliance_results.xlsx")
        self._save_statistics(results)
        logger.info("✅ All outputs generated")

    def _save_json(self, results, filename):
        fp = self.output_dir / filename
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info(f"  ✓ {fp} ({len(results)} records)")

    def _save_excel(self, results, filename):
        fp = self.output_dir / filename
        rows = [{
            'Conversation ID': r.conversation_id,
            'Compliant': 'YES' if r.compliant else 'NO',
            'Confidence': f"{r.confidence:.3f}",
            'Threshold': f"{r.threshold:.2f}",
            'Needs LLM Review': 'YES' if r.needs_llm_review else 'NO',
            'Violations': ', '.join(v.rule_id for v in r.violations) or '-',
            'Max Severity': max((v.severity for v in r.violations), default='-'),
            'Similarity Scores': str(r.similarity_scores),
            'Agent Messages': r.agent_message_count,
            'Evidence': '; '.join(r.evidence) or '-'
        } for r in results]

        df = pd.DataFrame(rows)
        with pd.ExcelWriter(fp, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Compliance Results', index=False)
            ws = writer.sheets['Compliance Results']
            for col in ws.columns:
                col = list(col)
                width = min(max(len(str(c.value or '')) for c in col) + 2, 50)
                ws.column_dimensions[col[0].column_letter].width = width
        logger.info(f"  ✓ {fp} (Excel)")

    def _save_statistics(self, results):
        total = len(results)
        auto = sum(1 for r in results if not r.needs_llm_review)
        review = sum(1 for r in results if r.needs_llm_review)
        compliant = sum(1 for r in results if r.compliant and not r.needs_llm_review)
        violations_count = sum(1 for r in results if not r.compliant and not r.needs_llm_review)

        severity_counts = defaultdict(int)
        for r in results:
            if not r.needs_llm_review:
                for v in r.violations:
                    severity_counts[v.severity] += 1

        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_conversations': total,
            'auto_decided': {
                'count': auto,
                'percentage': round(auto / total * 100, 1) if total > 0 else 0,
                'compliant': compliant,
                'violations': violations_count
            },
            'llm_review_needed': {
                'count': review,
                'percentage': round(review / total * 100, 1) if total > 0 else 0
            },
            'violations_by_severity': dict(severity_counts),
            'automation_rate': round(auto / total * 100, 1)
        }

        fp = self.output_dir / "statistics.json"
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  ✓ {fp}")


def main():
    print("="*70)
    print("LAYER 1: SEMANTIC COMPLIANCE CHECKER (FINE TUNED)")
    print("="*70 + "\n")

    if not Path("data/conversations.json").exists():
        print("❌ data/conversations.json not found")
        return

    with open("data/conversations.json", 'r') as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations\n")

    checker = SemanticComplianceChecker()
    results = checker.check_multiple(conversations)
    OutputGenerator().save_all(results)

    total = len(results)
    auto = sum(1 for r in results if not r.needs_llm_review)
    review = sum(1 for r in results if r.needs_llm_review)
    compliant = sum(1 for r in results if r.compliant and not r.needs_llm_review)
    violations_count = sum(1 for r in results if not r.compliant and not r.needs_llm_review)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"\n📊 Results:")
    print(f"  Total:         {total}")
    print(f"  Auto-decided:  {auto} ({auto/total*100:.1f}%)")
    print(f"  LLM review:    {review} ({review/total*100:.1f}%)")
    print(f"\n✅ Auto-decided:")
    print(f"  Compliant:     {compliant}")
    print(f"  Violations:    {violations_count}")
    print(f"\n📁 Outputs in data/layer1_output/\n{'='*70}")


if __name__ == "__main__":
    main()