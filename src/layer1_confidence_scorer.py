"""
Layer 1: Semantic Compliance Checker with Confidence Scoring

Uses sentence embeddings (Sentence-BERT) for semantic similarity matching
instead of pure regex/keywords. This catches violations even when the
exact words aren't used.

Combines:
1. Sentence embeddings for semantic matching
2. Keyword matching for known clear violations
3. Heuristic signals for confidence scoring
"""

import json
import re
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ComplianceResult:
    """Result from rule-based compliance check."""
    conversation_id: str
    compliant: bool
    confidence: float
    violations: List[Dict]
    evidence: List[str]
    matched_rules: List[str]
    similarity_scores: Dict[str, float]
    notes: str


class SemanticComplianceChecker:
    """
    Layer 1: Semantic classifier with confidence scoring.
    
    Uses sentence embeddings (SBERT) to detect violations based on
    semantic similarity to known violation examples, not just keywords.
    """
    
    def __init__(self, rules_file: str = "data/compliance_rules.json"):
        """Load compliance rules and initialize embedding model."""
        print("Loading compliance rules...")
        with open(rules_file, 'r') as f:
            rules_data = json.load(f)
        
        self.rules = {rule['id']: rule for rule in rules_data['rules']}
        self.rule_set_id = rules_data['rule_set_id']
        self.version = rules_data['version']
        
        # Load sentence embedding model (lightweight, runs on CPU)
        print("Loading sentence embedding model (this may take a moment)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # This model: 384 dimensions, fast inference, good quality
        # Alternative: 'all-mpnet-base-v2' (768 dim, slower but more accurate)
        
        # Build violation example embeddings
        self._build_violation_examples()
        
        # Build keyword patterns (still useful for clear cases)
        self._build_keyword_patterns()
        
        print("✅ Semantic checker initialized!")
    
    def _build_violation_examples(self):
        """
        Create example sentences for each violation type.
        We'll embed these and compare conversation text to them.
        """
        
        # Examples of what each violation looks like in practice
        self.violation_examples = {
            'R001': [  # Legal threats, garnishment
                "We're going to sue you if you don't pay",
                "I'll have your wages garnished",
                "We're sending this to our lawyers",
                "You'll be taken to court over this",
                "Legal action will be taken against you",
                "We can have you arrested for non-payment"
            ],
            'R002': [  # Contact family/employer
                "We'll contact your family about this debt",
                "Your employer will be notified",
                "We're going to call your boss",
                "Your family will find out about this",
                "We'll tell everyone you know"
            ],
            'R003': [  # Abusive/demeaning language
                "That's not our problem, figure it out yourself",
                "This is your fault for not paying",
                "I don't care about your excuses",
                "That's not my problem, deal with it",
                "You have to find a way, that's on you"
            ],
            'R004': [  # Immediate payment demands
                "You need to pay immediately",
                "Pay right now or face consequences",
                "We need payment in full by tomorrow",
                "You must pay this today",
                "Pay immediately, no exceptions"
            ],
            'R005': [  # Misrepresentation of consequences
                "Your credit will be ruined forever",
                "You'll never get a loan again",
                "We're going to destroy your credit rating",
                "This will ruin your financial future",
                "You'll never be able to buy a house"
            ]
        }
        
        # Examples of COMPLIANT language (for R006 - fair engagement)
        self.compliant_examples = [
            "I understand your situation, let me help",
            "I'm sorry to hear that, let me look into this",
            "We can work with you on a payment plan",
            "Let me escalate this to the right team",
            "I'll review your account and see what options are available",
            "I appreciate your patience while we resolve this"
        ]
        
        # Embed all violation examples
        print("Computing embeddings for violation examples...")
        self.violation_embeddings = {}
        for rule_id, examples in self.violation_examples.items():
            self.violation_embeddings[rule_id] = self.embedding_model.encode(
                examples,
                convert_to_numpy=True
            )
        
        # Embed compliant examples
        self.compliant_embeddings = self.embedding_model.encode(
            self.compliant_examples,
            convert_to_numpy=True
        )
        
        print(f"✅ Embedded {sum(len(e) for e in self.violation_examples.values())} violation examples")
    
    def _build_keyword_patterns(self):
        """Build keyword sets for fast exact matching (still useful)."""
        self.rule_keywords = {}
        
        for rule_id, rule in self.rules.items():
            if rule.get('keywords'):
                self.rule_keywords[rule_id] = set(
                    kw.lower() for kw in rule['keywords']
                )
    
    def check_conversation(self, conversation: Dict) -> ComplianceResult:
        """
        Analyze a conversation using semantic similarity.
        
        For each agent message:
        1. Embed the message
        2. Compare to violation example embeddings
        3. High similarity = likely violation
        4. Calculate confidence based on similarity scores
        """
        conv_id = conversation['conversation_id']
        
        # Extract agent messages
        agent_messages = [
            msg['text'] for msg in conversation['messages'] 
            if msg['role'] == 'agent'
        ]
        
        if not agent_messages:
            return ComplianceResult(
                conversation_id=conv_id,
                compliant=True,
                confidence=1.0,
                violations=[],
                evidence=[],
                matched_rules=[],
                similarity_scores={},
                notes="No agent messages found"
            )
        
        # Embed agent messages
        agent_embeddings = self.embedding_model.encode(
            agent_messages,
            convert_to_numpy=True
        )
        
        # Check for violations using semantic similarity
        violations = []
        evidence_list = []
        similarity_scores = {}
        
        for rule_id, rule_embeddings in self.violation_embeddings.items():
            if rule_id == 'R006':  # Skip positive rule for now
                continue
            
            # Calculate similarity between each agent message and violation examples
            max_similarity = 0.0
            best_match_msg = None
            best_match_idx = None
            
            for i, agent_emb in enumerate(agent_embeddings):
                # Compare this message to all examples of this violation
                similarities = cosine_similarity(
                    agent_emb.reshape(1, -1),
                    rule_embeddings
                )[0]
                
                msg_max_sim = similarities.max()
                if msg_max_sim > max_similarity:
                    max_similarity = msg_max_sim
                    best_match_msg = agent_messages[i]
                    best_match_idx = similarities.argmax()
            
            similarity_scores[rule_id] = round(float(max_similarity), 3)
            
            # Thresholds for semantic similarity
            # Higher = more strict (fewer false positives)
            # Lower = more sensitive (catches more potential violations)
            threshold = 0.60  # Tune this based on testing
            
            if max_similarity >= threshold:
                # Also check keywords for confirmation (hybrid approach)
                keyword_match = False
                if rule_id in self.rule_keywords:
                    for keyword in self.rule_keywords[rule_id]:
                        if keyword in best_match_msg.lower():
                            keyword_match = True
                            break
                
                # Higher confidence if both semantic AND keyword match
                if keyword_match or max_similarity >= 0.75:
                    violations.append({
                        'rule_id': rule_id,
                        'severity': self.rules[rule_id]['severity'],
                        'category': self.rules[rule_id]['category'],
                        'description': self.rules[rule_id]['description'],
                        'similarity': max_similarity,
                        'keyword_match': keyword_match
                    })
                    
                    example_matched = self.violation_examples[rule_id][best_match_idx]
                    evidence_list.append(
                        f"{rule_id} (sim: {max_similarity:.2f}): \"{best_match_msg[:80]}...\" "
                        f"(similar to: \"{example_matched}\")"
                    )
        
        # Check for positive engagement (R006)
        has_positive_engagement = self._check_positive_engagement(
            agent_embeddings,
            agent_messages
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            violations=violations,
            similarity_scores=similarity_scores,
            has_positive_engagement=has_positive_engagement,
            num_messages=len(agent_messages),
            agent_messages=agent_messages
        )
        
        compliant = len(violations) == 0
        matched_rules = [v['rule_id'] for v in violations]
        
        notes = self._generate_notes(
            violations,
            has_positive_engagement,
            len(agent_messages),
            similarity_scores
        )
        
        return ComplianceResult(
            conversation_id=conv_id,
            compliant=compliant,
            confidence=confidence,
            violations=violations,
            evidence=evidence_list,
            matched_rules=matched_rules,
            similarity_scores=similarity_scores,
            notes=notes
        )
    
    def _check_positive_engagement(self, agent_embeddings: np.ndarray,
                                   agent_messages: List[str]) -> bool:
        """Check if agent used empathetic language (R006)."""
        # Compare agent messages to compliant examples
        for agent_emb in agent_embeddings:
            similarities = cosine_similarity(
                agent_emb.reshape(1, -1),
                self.compliant_embeddings
            )[0]
            
            if similarities.max() >= 0.55:  # Found empathetic language
                return True
        
        return False
    
    def _calculate_confidence(self, violations: List[Dict],
                             similarity_scores: Dict[str, float],
                             has_positive_engagement: bool,
                             num_messages: int,
                             agent_messages: List[str]) -> float:
        """
        Calculate confidence score using semantic similarity scores.
        
        Key insight: High semantic similarity = high confidence
        """
        
        if len(violations) == 0:
            # NO violations detected
            base_confidence = 0.55
            
            # Check if we're VERY dissimilar to all violations
            avg_similarity = np.mean(list(similarity_scores.values()))
            if avg_similarity < 0.40:
                # Very different from violations = very confident it's compliant
                base_confidence += 0.20
            elif avg_similarity < 0.50:
                base_confidence += 0.10
            
            # Positive engagement boost
            if has_positive_engagement:
                base_confidence += 0.15
            
        else:
            # Violations detected
            base_confidence = 0.50
            
            # Confidence based on similarity strength
            max_violation_sim = max(v['similarity'] for v in violations)
            
            if max_violation_sim >= 0.80:
                # Very similar to known violations = very confident
                base_confidence += 0.30
            elif max_violation_sim >= 0.70:
                base_confidence += 0.20
            elif max_violation_sim >= 0.60:
                base_confidence += 0.10
            
            # Keyword confirmation bonus
            has_keyword_match = any(v.get('keyword_match', False) for v in violations)
            if has_keyword_match:
                base_confidence += 0.15
            
            # Multiple violations
            if len(violations) >= 2:
                base_confidence += 0.10
            
            # Severity boost
            max_severity = max(v['severity'] for v in violations)
            if max_severity == 'critical':
                base_confidence += 0.15
            elif max_severity == 'high':
                base_confidence += 0.10
        
        # Check for ambiguous phrases (still useful)
        full_text = " ".join(agent_messages).lower()
        ambiguous_phrases = [
            'time is of the essence', 'we really need', 'strongly encourage',
            'as soon as possible', 'urgently', 'you should really'
        ]
        ambiguous_count = sum(1 for phrase in ambiguous_phrases if phrase in full_text)
        if ambiguous_count > 0:
            base_confidence -= (ambiguous_count * 0.08)
        
        # Message count factor
        if num_messages >= 3:
            base_confidence += 0.05
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, base_confidence))
        return round(confidence, 2)
    
    def _generate_notes(self, violations: List[Dict],
                       has_positive: bool,
                       num_messages: int,
                       similarity_scores: Dict[str, float]) -> str:
        """Generate notes about the analysis."""
        if not violations:
            avg_sim = np.mean(list(similarity_scores.values()))
            if has_positive:
                return f"No violations (avg similarity to violations: {avg_sim:.2f}). Agent used empathetic language."
            else:
                return f"No violations (avg similarity to violations: {avg_sim:.2f}), but agent could be more empathetic."
        
        violation_summary = ", ".join(
            f"{v['rule_id']}(sim:{v['similarity']:.2f})"
            for v in violations
        )
        return f"Violations detected: {violation_summary}. {num_messages} messages analyzed."
    
    def get_threshold(self, result: ComplianceResult) -> float:
        """Get confidence threshold based on severity."""
        if result.compliant:
            return 0.70
        
        if not result.violations:
            return 0.75
        
        max_severity = max(v['severity'] for v in result.violations)
        
        return {
            'low': 0.70,
            'medium': 0.75,
            'high': 0.80,
            'critical': 0.85
        }.get(max_severity, 0.75)
    
    def needs_llm_review(self, result: ComplianceResult) -> bool:
        """Check if this needs LLM review."""
        threshold = self.get_threshold(result)
        return result.confidence < threshold


def main():
    """Test the semantic checker."""
    import os
    
    if not os.path.exists("data/generated_conversations.json"):
        print("❌ Error: data/generated_conversations.json not found")
        print("Please run: python generate_conversations.py")
        return
    
    print("Loading conversations...")
    with open("data/generated_conversations.json", 'r') as f:
        conversations = json.load(f)
    
    print("\nInitializing semantic checker...")
    print("(First run will download the model, ~90MB)")
    checker = SemanticComplianceChecker()
    
    print(f"\n{'='*70}")
    print(f"Analyzing {len(conversations)} conversations with SEMANTIC matching")
    print(f"{'='*70}\n")
    
    results = []
    stats = {
        'high_conf_compliant': 0,
        'high_conf_violation': 0,
        'low_conf_needs_llm': 0,
        'by_severity': defaultdict(int)
    }
    
    # Analyze first 10 for demo
    for conv in conversations[:10]:
        result = checker.check_conversation(conv)
        results.append(result)
        
        threshold = checker.get_threshold(result)
        if result.confidence >= threshold:
            if result.compliant:
                stats['high_conf_compliant'] += 1
            else:
                stats['high_conf_violation'] += 1
                for v in result.violations:
                    stats['by_severity'][v['severity']] += 1
        else:
            stats['low_conf_needs_llm'] += 1
        
        needs_llm = checker.needs_llm_review(result)
        status = "✅ COMPLIANT" if result.compliant else "❌ VIOLATION"
        llm_flag = " → 🤖 LLM Review" if needs_llm else " → ✓ Auto-Decide"
        
        print(f"{result.conversation_id}: {status} (conf: {result.confidence:.2f}){llm_flag}")
        print(f"  Threshold: {threshold:.2f}")
        
        if result.violations:
            for v in result.violations:
                keyword_flag = "✓" if v.get('keyword_match') else ""
                print(f"  • {v['rule_id']} ({v['severity']}) - similarity: {v['similarity']:.2f} {keyword_flag}")
        
        # Show top similarity scores
        top_sims = sorted(result.similarity_scores.items(), 
                         key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top similarities: {', '.join(f'{r}:{s:.2f}' for r, s in top_sims)}")
        
        if result.evidence:
            print(f"  Evidence: {result.evidence[0][:100]}...")
        
        print()
    
    print("=" * 70)
    print("SUMMARY (first 10):")
    print(f"  ✅ High confidence compliant: {stats['high_conf_compliant']}")
    print(f"  ❌ High confidence violations: {stats['high_conf_violation']}")
    print(f"  🤖 Low confidence (needs LLM): {stats['low_conf_needs_llm']}")
    
    if stats['by_severity']:
        print(f"\n  Violations by severity:")
        for severity, count in stats['by_severity'].items():
            print(f"    {severity}: {count}")
    
    print(f"\n✅ Semantic Layer 1 complete!")
    print(f"💡 This uses sentence embeddings (SBERT) to catch semantic violations")
    print(f"   not just exact keyword matches!")


if __name__ == "__main__":
    main()