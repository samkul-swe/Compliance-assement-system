"""
Layer 1: Semantic Compliance Checker (Centroid + Learned Boundaries)

Architecture:
    Step 1 — Binary check: is this message in violation space?
             Compare against combined violation centroid vs compliant centroid

    Step 2 — Severity classification via ThresholdLearner:
             Learned min/max boundaries per severity from training examples
             Score falls in a range → that severity
             Score near boundary → Layer 2

Uses base SBERT model (all-MiniLM-L6-v2)
ThresholdLearner learns boundaries from SEVERITY_EXAMPLES at startup
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from threshold_learner import ThresholdLearner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


SEVERITY_EXAMPLES = {
    'critical': [
        "We're going to sue you if you don't pay",
        "I'll have your wages garnished",
        "We're sending this to our lawyers",
        "You'll be taken to court over this",
        "Legal action will be taken against you",
        "We can have you arrested for non-payment",
        "Our legal department will be in touch shortly",
        "This will be passed to our litigation team",
        "You'll be hearing from our solicitors",
        "A county court judgment will be filed against you",
        "We will begin legal proceedings immediately",
        "You can be imprisoned for ignoring this debt",
        "We'll contact your family about this debt",
        "Your employer will be notified",
        "We're going to call your boss",
        "Your family will find out about this",
        "We'll tell everyone you know",
        "Your workplace will receive a letter from us",
        "We'll be speaking to your next of kin",
        "Everyone on your application will be contacted",
        "Your spouse will be informed about this debt",
        "We have your employer's details and will be in touch",
    ],
    'high': [
        "That's not our problem, figure it out yourself",
        "This is your fault for not paying",
        "I don't care about your excuses",
        "That's not my problem, deal with it",
        "You have to find a way, that's on you",
        "Stop wasting my time with these stories",
        "Everyone has problems, you still owe the money",
        "You're being deliberately difficult about this",
        "I find your attitude quite unacceptable",
        "Other customers manage to pay, why can't you",
        "Your credit will be ruined forever",
        "You'll never get a loan again",
        "We're going to destroy your credit rating",
        "This will ruin your financial future",
        "You'll never be able to buy a house",
        "No bank will ever lend to you after this",
        "Your credit file will be permanently damaged",
        "You'll be blacklisted from every financial institution",
        "This debt means you'll never get approved for anything",
        "Your financial life is essentially over if this isn't paid",
    ],
    'medium': [
        "You need to pay immediately",
        "Pay right now or face consequences",
        "We need payment in full by tomorrow",
        "You must pay this today",
        "Pay immediately, no exceptions",
        "The full balance must be cleared today",
        "Payment in full is required before this call ends",
        "We cannot accept anything less than full payment now",
        "You have to clear this immediately, no options",
        "We require same-day payment of the full amount",
        "That's not relevant to what we're discussing",
        "Your personal circumstances don't change what you owe",
        "We can't consider personal situations, the debt is the debt",
        "That's unfortunate but it doesn't affect the balance",
        "Personal issues aren't something we can factor in",
        "We deal with the account, not personal circumstances",
        "Whatever is happening in your life, the debt still stands",
        "That doesn't affect our position on this account",
        "I can't make decisions based on personal circumstances",
        "Regardless of your situation, the amount remains outstanding",
    ],
    'low': [
        "I really do need you to commit to something today",
        "It would be really in your best interest to sort this now",
        "I'd strongly encourage you to make a payment today",
        "The sooner you pay the better it will be for you",
        "You really should take care of this as soon as possible",
        "I think you need to prioritise this above other things",
        "It's really important that we get something agreed today",
        "You don't want this to get any worse than it already is",
        "I'd really like to get this resolved on today's call",
        "The longer this goes on the more difficult it becomes",
        "You really need to take this more seriously",
        "I'd urge you to make a decision before we finish today",
        "Things can get complicated if we don't sort this today",
        "It's really best for you if we agree something right now",
        "I think you should seriously consider paying something today",
        "You really can't afford to leave this any longer",
        "I'd recommend you think very carefully about delaying this",
        "It's quite urgent that we get a resolution today",
        "You should really try to sort something out before we hang up",
        "I'd be very cautious about leaving this unresolved today",
    ],
    'compliant': [
        "I completely understand this is a really difficult situation",
        "I'm sorry to hear you're going through such a tough time",
        "Thank you for letting me know what's happening for you",
        "I can hear that this is causing you a lot of stress",
        "I really appreciate you being so open with me today",
        "Let me look at what options we have available for you",
        "There are a few different ways we could approach this",
        "I'd like to explore what might work best for your situation",
        "We do have hardship arrangements that might help you",
        "Let me see what flexibility we have on your account",
        "I can look at setting up a payment plan that suits you",
        "Take all the time you need, there's no rush on this call",
        "We don't need to resolve everything today if that's too much",
        "Please don't feel pressured, we're just here to talk",
        "There's no obligation to decide anything right now",
        "I'll make sure to note all of this on your account",
        "Let me escalate this to our specialist team for you",
        "I want to make sure you're getting the right support",
        "I'm sorry to hear you had a problem with the product",
        "Let me look at whether the charges are valid given what happened",
    ]
}

SEVERITY_ORDER = ['compliant', 'low', 'medium', 'high', 'critical']


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
    detected_severity: Optional[str]
    severity_confidence: float
    needs_llm_review: bool
    sitting_between: Optional[str]
    violations: List[ComplianceViolation]
    evidence: List[str]
    agent_message_count: int
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            'conversation_id': self.conversation_id,
            'compliant': bool(self.compliant),
            'detected_severity': self.detected_severity,
            'severity_confidence': float(self.severity_confidence),
            'needs_llm_review': bool(self.needs_llm_review),
            'sitting_between': self.sitting_between,
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
            'agent_message_count': self.agent_message_count,
            'timestamp': self.timestamp
        }


class ConfigLoader:

    @staticmethod
    def load_compliance_rules(rules_file: str = "data/compliance_rules.json") -> Dict:
        path = Path(rules_file)
        if not path.exists():
            raise FileNotFoundError(f"Rules not found: {rules_file}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_config(config_file: str = "config/severity_confidence.json") -> Dict:
        path = Path(config_file)
        if not path.exists():
            return {'boundary_margin': 0.04}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


class SemanticComplianceChecker:

    def __init__(self,
                 rules_file: str = "data/compliance_rules.json",
                 config_file: str = "config/severity_confidence.json"):

        logger.info("Initializing Compliance Checker...")

        self.rules_data = ConfigLoader.load_compliance_rules(rules_file)
        self.config = ConfigLoader.load_config(config_file)
        self.rules = {r['id']: r for r in self.rules_data['rules']}
        self.severity_rank = {s: i for i, s in enumerate(SEVERITY_ORDER)}

        logger.info("Loading SBERT model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded ✓")

        # Learn severity ranges from training examples
        # Each phrase gets scored against violation pool
        # Min/max per severity = learned range on compliance spectrum
        self.threshold_learner = ThresholdLearner(
            model=self.model,
            severity_examples=SEVERITY_EXAMPLES
        )
        self.threshold_learner.save_ranges()

        logger.info("✅ Checker initialized\n")

    def check_conversation(self, conversation: Dict) -> ComplianceResult:
        conv_id = conversation['conversation_id']
        boundary_margin = self.config.get('boundary_margin', 0.04)

        agent_messages = [
            (i, msg['text'])
            for i, msg in enumerate(conversation['messages'])
            if msg['role'] == 'agent'
        ]

        if not agent_messages:
            return ComplianceResult(
                conversation_id=conv_id,
                compliant=True,
                detected_severity=None,
                severity_confidence=1.0,
                needs_llm_review=False,
                sitting_between=None,
                violations=[],
                evidence=[],
                agent_message_count=0,
                timestamp=datetime.now().isoformat()
            )

        agent_texts = [text for _, text in agent_messages]
        agent_embeddings = self.model.encode(agent_texts, convert_to_numpy=True)

        violations = []
        evidence_list = []

        for (orig_idx, msg_text), agent_emb in zip(agent_messages, agent_embeddings):

            # ── Step 1 + Step 2 combined via ThresholdLearner ──
            # classify() finds the closest severity centroid directly
            # If closest is compliant → not a violation → skip
            detected_severity, best_score, sitting_between = \
                self.threshold_learner.classify(agent_emb, boundary_margin)

            if detected_severity == 'compliant':
                continue

            # Match rule to detected severity
            matching_rules = [
                r for r in self.rules.values()
                if r['severity'] == detected_severity
            ]
            matched_rule = matching_rules[0] if matching_rules \
                else list(self.rules.values())[0]

            violations.append(ComplianceViolation(
                rule_id=matched_rule['id'],
                category=matched_rule['category'],
                severity=detected_severity,
                description=matched_rule['description'],
                message_index=orig_idx,
                matched_text=msg_text[:100],
                similarity_score=float(best_score),
            ))

            evidence_list.append(
                f"{detected_severity.upper()}: \"{msg_text[:80]}...\" "
                f"(score: {best_score:.3f}"
                + (f", near boundary: {sitting_between}" if sitting_between else "")
                + ")"
            )

        if not violations:
            return ComplianceResult(
                conversation_id=conv_id,
                compliant=True,
                detected_severity=None,
                severity_confidence=1.0,
                needs_llm_review=False,
                sitting_between=None,
                violations=[],
                evidence=[],
                agent_message_count=len(agent_messages),
                timestamp=datetime.now().isoformat()
            )

        # Take worst violation
        worst = max(violations, key=lambda v: self.severity_rank.get(v.severity, 0))

        # Re-classify worst to get sitting_between for escalation decision
        worst_emb = self.model.encode([worst.matched_text], convert_to_numpy=True)[0]
        _, _, sitting_between = self.threshold_learner.classify(
            worst_emb, boundary_margin
        )

        needs_review = sitting_between is not None

        return ComplianceResult(
            conversation_id=conv_id,
            compliant=False,
            detected_severity=worst.severity,
            severity_confidence=float(worst.similarity_score),
            needs_llm_review=needs_review,
            sitting_between=sitting_between,
            violations=violations,
            evidence=evidence_list,
            agent_message_count=len(agent_messages),
            timestamp=datetime.now().isoformat()
        )

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
        auto   = [r for r in results if not r.needs_llm_review]
        review = [r for r in results if r.needs_llm_review]
        self._save_json(auto,   "auto_decided.json")
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
            'Conversation ID':    r.conversation_id,
            'Compliant':          'YES' if r.compliant else 'NO',
            'Detected Severity':  r.detected_severity or '-',
            'Severity Score':     f"{r.severity_confidence:.3f}",
            'Needs LLM Review':   'YES' if r.needs_llm_review else 'NO',
            'Sitting Between':    r.sitting_between or '-',
            'Violations':         ', '.join(f"{v.rule_id}({v.severity})"
                                            for v in r.violations) or '-',
            'Agent Messages':     r.agent_message_count,
            'Evidence':           '; '.join(r.evidence) or '-'
        } for r in results]

        df = pd.DataFrame(rows)
        with pd.ExcelWriter(fp, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Compliance Results', index=False)
            ws = writer.sheets['Compliance Results']
            for col in ws.columns:
                col = list(col)
                w = min(max(len(str(c.value or '')) for c in col) + 2, 60)
                ws.column_dimensions[col[0].column_letter].width = w
        logger.info(f"  ✓ {fp} (Excel)")

    def _save_statistics(self, results):
        total = len(results)
        if total == 0:
            return
        auto      = sum(1 for r in results if not r.needs_llm_review)
        review    = sum(1 for r in results if r.needs_llm_review)
        compliant = sum(1 for r in results if r.compliant)
        violations = sum(1 for r in results if not r.compliant)

        sev_counts = defaultdict(int)
        for r in results:
            if r.detected_severity:
                sev_counts[r.detected_severity] += 1

        stats = {
            'timestamp':    datetime.now().isoformat(),
            'total':        total,
            'compliant':    compliant,
            'violations':   violations,
            'auto_decided': {'count': auto,   'pct': round(auto/total*100, 1)},
            'llm_review':   {'count': review, 'pct': round(review/total*100, 1)},
            'by_severity':  dict(sev_counts)
        }

        fp = self.output_dir / "statistics.json"
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  ✓ {fp}")


def main():
    print("=" * 70)
    print("LAYER 1: CENTROID + LEARNED BOUNDARY COMPLIANCE CHECKER")
    print("=" * 70 + "\n")

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
    if total == 0:
        print("No results.")
        return

    auto      = sum(1 for r in results if not r.needs_llm_review)
    review    = sum(1 for r in results if r.needs_llm_review)
    compliant = sum(1 for r in results if r.compliant)
    violations = sum(1 for r in results if not r.compliant)

    sev_counts = defaultdict(int)
    for r in results:
        if r.detected_severity:
            sev_counts[r.detected_severity] += 1

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"\n📊 Results:")
    print(f"  Total:      {total}")
    print(f"  Compliant:  {compliant} ({compliant/total*100:.1f}%)")
    print(f"  Violations: {violations} ({violations/total*100:.1f}%)")
    print(f"\n🎯 Severity Breakdown:")
    for sev in ['critical', 'high', 'medium', 'low']:
        print(f"  {sev:10}: {sev_counts.get(sev, 0)}")
    print(f"\n⚡ Escalation:")
    print(f"  Auto-decided:  {auto} ({auto/total*100:.1f}%)")
    print(f"  Layer 2 queue: {review} ({review/total*100:.1f}%)")
    print(f"\n📁 Outputs in data/layer1_output/\n{'='*70}")


if __name__ == "__main__":
    main()