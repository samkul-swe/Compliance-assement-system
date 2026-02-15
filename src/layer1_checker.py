"""
Layer 1: Semantic Compliance Checker

Uses sentence embeddings (SBERT) to detect violations based on semantic similarity.
Simple severity-based confidence thresholds from JSON.

Features:
- Semantic matching (catches paraphrased violations)
- Configurable confidence thresholds by severity
- Outputs to JSON and Excel
- Separates high-confidence decisions from low-confidence (needs LLM review)
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ComplianceViolation:
    """Single violation detected."""
    rule_id: str
    category: str
    severity: str
    description: str
    message_index: int
    matched_text: str
    similarity_score: float
    keyword_match: bool


@dataclass
class ComplianceResult:
    """Complete result for one conversation."""
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
        """Convert to dictionary for JSON serialization."""
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
                    'keyword_match': v.keyword_match
                }
                for v in self.violations
            ],
            'evidence': self.evidence,
            'similarity_scores': {k: float(v) for k, v in self.similarity_scores.items()},
            'agent_message_count': self.agent_message_count,
            'timestamp': self.timestamp
        }


class ConfigLoader:
    """Load configuration from JSON files."""
    
    @staticmethod
    def load_compliance_rules(rules_file: str = "data/compliance_rules.json") -> Dict:
        """Load compliance rules."""
        path = Path(rules_file)
        if not path.exists():
            raise FileNotFoundError(f"Compliance rules not found: {rules_file}")
        
        with open(path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        # Validate against schema if available
        schema_path = Path("docs/api/compliance_rules_schema.json")
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            # Basic validation that required fields exist
            if 'rules' not in rules_data:
                raise ValueError("Rules data doesn't match schema")
        
        return rules_data
    
    @staticmethod
    def load_confidence_config(config_file: str = "config/severity_confidence.json") -> Dict:
        """Load confidence threshold configuration."""
        path = Path(config_file)
        if not path.exists():
            logger.warning(f"Config file not found: {config_file}, using defaults")
            return ConfigLoader._default_config()
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def _default_config() -> Dict:
        """Default configuration if file doesn't exist."""
        return {
            "semantic_threshold": 0.60,
            "confidence_thresholds": {
                "compliant": 0.70,
                "low": 0.70,
                "medium": 0.75,
                "high": 0.80,
                "critical": 0.85
            }
        }


class SemanticComplianceChecker:
    """
    Layer 1: Semantic compliance checker with simple severity-based thresholds.
    """
    
    def __init__(self, 
                 rules_file: str = "data/compliance_rules.json",
                 config_file: str = "config/severity_confidence.json"):
        """Initialize checker with rules and configuration."""
        logger.info("Initializing Semantic Compliance Checker...")
        
        # Load configuration
        self.rules_data = ConfigLoader.load_compliance_rules(rules_file)
        self.config = ConfigLoader.load_confidence_config(config_file)
        
        self.rules = {rule['id']: rule for rule in self.rules_data['rules']}
        logger.info(f"Loaded {len(self.rules)} compliance rules")
        
        # Load embedding model
        logger.info("Loading sentence embedding model (this may take a moment)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
        
        # Build violation examples
        self._build_examples()
        
        # Build keyword patterns
        self._build_keyword_patterns()
        
        logger.info("✅ Semantic checker initialized\n")
    
    def _build_examples(self):
        """Build example sentences for semantic matching."""
        # Violation examples for each rule
        self.violation_examples = {
            'R001': [
                "We're going to sue you if you don't pay",
                "I'll have your wages garnished",
                "We're sending this to our lawyers",
                "You'll be taken to court over this",
                "Legal action will be taken against you",
                "We can have you arrested for non-payment"
            ],
            'R002': [
                "We'll contact your family about this debt",
                "Your employer will be notified",
                "We're going to call your boss",
                "Your family will find out about this",
                "We'll tell everyone you know"
            ],
            'R003': [
                "That's not our problem, figure it out yourself",
                "This is your fault for not paying",
                "I don't care about your excuses",
                "That's not my problem, deal with it",
                "You have to find a way, that's on you"
            ],
            'R004': [
                "You need to pay immediately",
                "Pay right now or face consequences",
                "We need payment in full by tomorrow",
                "You must pay this today",
                "Pay immediately, no exceptions"
            ],
            'R005': [
                "Your credit will be ruined forever",
                "You'll never get a loan again",
                "We're going to destroy your credit rating",
                "This will ruin your financial future",
                "You'll never be able to buy a house"
            ]
        }
        
        # Embed all examples
        logger.info("Computing embeddings for violation examples...")
        self.violation_embeddings = {}
        for rule_id, examples in self.violation_examples.items():
            self.violation_embeddings[rule_id] = self.embedding_model.encode(
                examples,
                convert_to_numpy=True
            )
        
        logger.info(f"Embedded {sum(len(e) for e in self.violation_examples.values())} violation examples")
    
    def _build_keyword_patterns(self):
        """Build keyword sets for exact matching."""
        self.rule_keywords = {}
        for rule_id, rule in self.rules.items():
            if rule.get('keywords'):
                self.rule_keywords[rule_id] = set(
                    kw.lower() for kw in rule['keywords']
                )
    
    def check_conversation(self, conversation: Dict) -> ComplianceResult:
        """
        Analyze a conversation for compliance violations.
        
        Returns ComplianceResult with all details.
        """
        conv_id = conversation['conversation_id']
        
        # Extract agent messages
        agent_messages = [
            (i, msg['text']) for i, msg in enumerate(conversation['messages']) 
            if msg['role'] == 'agent'
        ]
        
        if not agent_messages:
            return ComplianceResult(
                conversation_id=conv_id,
                compliant=True,
                confidence=1.0,
                threshold=0.70,
                needs_llm_review=False,
                violations=[],
                evidence=[],
                similarity_scores={},
                agent_message_count=0,
                timestamp=datetime.now().isoformat()
            )
        
        # Embed agent messages
        agent_texts = [text for _, text in agent_messages]
        agent_embeddings = self.embedding_model.encode(
            agent_texts,
            convert_to_numpy=True
        )
        
        # Check for violations
        violations = []
        evidence_list = []
        similarity_scores = {}
        
        semantic_threshold = self.config['semantic_threshold']
        
        for rule_id, rule_embeddings in self.violation_embeddings.items():
            if rule_id not in self.rules:
                continue
            
            max_similarity = 0.0
            best_match_msg_idx = None
            best_match_text = None
            
            for (orig_idx, msg_text), agent_emb in zip(agent_messages, agent_embeddings):
                # Compare to all examples of this violation
                similarities = cosine_similarity(
                    agent_emb.reshape(1, -1),
                    rule_embeddings
                )[0]
                
                msg_max_sim = similarities.max()
                if msg_max_sim > max_similarity:
                    max_similarity = msg_max_sim
                    best_match_msg_idx = orig_idx
                    best_match_text = msg_text
            
            similarity_scores[rule_id] = round(float(max_similarity), 3)
            
            # Check if exceeds threshold
            if max_similarity >= semantic_threshold:
                # Check keywords for confirmation
                keyword_match = False
                if rule_id in self.rule_keywords:
                    for keyword in self.rule_keywords[rule_id]:
                        if keyword in best_match_text.lower():
                            keyword_match = True
                            break
                
                # High similarity or keyword match = violation
                if keyword_match or max_similarity >= 0.75:
                    rule = self.rules[rule_id]
                    violations.append(ComplianceViolation(
                        rule_id=rule_id,
                        category=rule['category'],
                        severity=rule['severity'],
                        description=rule['description'],
                        message_index=best_match_msg_idx,
                        matched_text=best_match_text[:100],
                        similarity_score=max_similarity,
                        keyword_match=keyword_match
                    ))
                    
                    evidence_list.append(
                        f"{rule_id} ({rule['severity']}): \"{best_match_text[:80]}...\" "
                        f"(similarity: {max_similarity:.2f})"
                    )
        
        # Calculate confidence (simple: based on max similarity)
        confidence = self._calculate_confidence(violations, similarity_scores)
        
        compliant = len(violations) == 0
        threshold = self._get_threshold(violations)
        needs_review = confidence < threshold
        
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
    
    def _calculate_confidence(self,
                            violations: List[ComplianceViolation],
                            similarity_scores: Dict[str, float]) -> float:
        """
        Calculate confidence score simply based on similarity.
        """
        if len(violations) == 0:
            # No violations - confidence based on how dissimilar to violations
            avg_similarity = np.mean(list(similarity_scores.values()))
            # Inverse relationship: lower similarity = higher confidence it's compliant
            confidence = 1.0 - (avg_similarity * 0.5)  # Scale down the impact
        else:
            # Violations detected - confidence based on max similarity
            max_similarity = max(v.similarity_score for v in violations)
            # Direct relationship: higher similarity = higher confidence it's a violation
            confidence = max_similarity
            
            # Boost if keyword match
            has_keyword = any(v.keyword_match for v in violations)
            if has_keyword:
                confidence = min(1.0, confidence + 0.10)
        
        return round(confidence, 2)
    
    def _get_threshold(self, violations: List[ComplianceViolation]) -> float:
        """Get confidence threshold based on severity."""
        thresholds = self.config['confidence_thresholds']
        
        if not violations:
            return thresholds['compliant']
        
        max_severity = max(v.severity for v in violations)
        return thresholds.get(max_severity, thresholds['medium'])
    
    def check_multiple(self, conversations: List[Dict]) -> List[ComplianceResult]:
        """Check multiple conversations."""
        results = []
        
        for i, conv in enumerate(conversations, 1):
            if i % 20 == 0:
                logger.info(f"Processed {i}/{len(conversations)} conversations...")
            
            try:
                result = self.check_conversation(conv)
                results.append(result)
            except Exception as e:
                logger.error(f"Error checking {conv.get('conversation_id')}: {e}")
        
        return results


class OutputGenerator:
    """Generate outputs in JSON and Excel formats."""
    
    def __init__(self, output_dir: str = "data/layer1_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_all(self, results: List[ComplianceResult]):
        """Save all outputs."""
        logger.info("\nGenerating outputs...")
        
        # 1. Auto-decided (high confidence) - JSON
        auto_decided = [r for r in results if not r.needs_llm_review]
        self._save_json(auto_decided, "auto_decided.json")
        
        # 2. LLM review queue - JSON
        llm_queue = [r for r in results if r.needs_llm_review]
        self._save_json(llm_queue, "llm_review_queue.json")
        
        # 3. Excel for human review (all results)
        self._save_excel(results, "compliance_results.xlsx")
        
        # 4. Statistics
        self._save_statistics(results)
        
        logger.info("✅ All outputs generated")
    
    def _save_json(self, results: List[ComplianceResult], filename: str):
        """Save results to JSON."""
        filepath = self.output_dir / filename
        
        data = [r.to_dict() for r in results]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"  ✓ {filepath} ({len(results)} records)")
    
    def _save_excel(self, results: List[ComplianceResult], filename: str):
        """Save results to Excel for human review."""
        filepath = self.output_dir / filename
        
        # Prepare data
        rows = []
        for r in results:
            rows.append({
                'Conversation ID': r.conversation_id,
                'Compliant': 'YES' if r.compliant else 'NO',
                'Confidence': f"{r.confidence:.2%}",
                'Threshold': f"{r.threshold:.2%}",
                'Violations': ', '.join([v.rule_id for v in r.violations]) if r.violations else '-',
                'Max Severity': max([v.severity for v in r.violations], default='-'),
                'Agent Messages': r.agent_message_count,
                'Evidence': '; '.join(r.evidence) if r.evidence else '-'
            })
        
        df = pd.DataFrame(rows)
        
        # Save with formatting
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Compliance Results', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Compliance Results']
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        logger.info(f"  ✓ {filepath} (Excel)")
    
    def _save_statistics(self, results: List[ComplianceResult]):
        """Save statistics."""
        total = len(results)
        auto_decided = sum(1 for r in results if not r.needs_llm_review)
        llm_needed = sum(1 for r in results if r.needs_llm_review)
        
        compliant = sum(1 for r in results if r.compliant and not r.needs_llm_review)
        violations = sum(1 for r in results if not r.compliant and not r.needs_llm_review)
        
        # Violations by severity
        severity_counts = defaultdict(int)
        for r in results:
            if not r.needs_llm_review:
                for v in r.violations:
                    severity_counts[v.severity] += 1
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_conversations': total,
            'auto_decided': {
                'count': auto_decided,
                'percentage': round(auto_decided / total * 100, 1),
                'compliant': compliant,
                'violations': violations
            },
            'llm_review_needed': {
                'count': llm_needed,
                'percentage': round(llm_needed / total * 100, 1)
            },
            'violations_by_severity': dict(severity_counts),
            'automation_rate': round(auto_decided / total * 100, 1)
        }
        
        filepath = self.output_dir / "statistics.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"  ✓ {filepath}")


def main():
    """Run Layer 1 compliance checker."""
    print("="*70)
    print("LAYER 1: SEMANTIC COMPLIANCE CHECKER")
    print("="*70 + "\n")
    
    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer
        import pandas as pd
    except ImportError:
        print("❌ Missing dependencies. Install with:")
        print("   pip install sentence-transformers pandas openpyxl scikit-learn")
        return
    
    # Check data files
    if not Path("data/conversations.json").exists():
        print("❌ data/conversations.json not found")
        print("   Run generate_conversations.py first")
        return
    
    # Load conversations
    print("Loading conversations...")
    with open("data/conversations.json", 'r') as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations\n")
    
    # Initialize checker
    checker = SemanticComplianceChecker()
    
    # Check all conversations
    print("="*70)
    print("ANALYZING CONVERSATIONS")
    print("="*70 + "\n")
    
    results = checker.check_multiple(conversations)
    
    # Generate outputs
    output_gen = OutputGenerator()
    output_gen.save_all(results)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total = len(results)
    auto_decided = sum(1 for r in results if not r.needs_llm_review)
    llm_needed = sum(1 for r in results if r.needs_llm_review)
    
    print(f"\n📊 Results:")
    print(f"  Total: {total}")
    print(f"  Auto-decided: {auto_decided} ({auto_decided/total*100:.1f}%)")
    print(f"  Needs LLM review: {llm_needed} ({llm_needed/total*100:.1f}%)")
    
    compliant = sum(1 for r in results if r.compliant and not r.needs_llm_review)
    violations = sum(1 for r in results if not r.compliant and not r.needs_llm_review)
    
    print(f"\n✅ Auto-decided breakdown:")
    print(f"  Compliant: {compliant}")
    print(f"  Violations: {violations}")
    
    print(f"\n📁 Outputs:")
    print(f"  data/layer1_output/")
    print(f"    ├─ auto_decided.json ({auto_decided} conversations) → JSON")
    print(f"    ├─ llm_review_queue.json ({llm_needed} conversations) → JSON for Layer 2")
    print(f"    ├─ layer1_results.xlsx (all {total} conversations) → Excel ⭐")
    print(f"    └─ statistics.json (metrics) → JSON")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()