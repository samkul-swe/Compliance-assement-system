"""
Layer 2: Severity Adjudication

Layer 1 has already confirmed a violation exists.
Layer 2's only job is to confirm which severity bucket it belongs to.

Receives from Layer 1:
    - conversation_id
    - detected_severity (best guess from centroid)
    - sitting_between   (e.g. "high and critical")
    - violations        (which messages triggered)
    - evidence          (centroid similarity scores)

Asks two LLM judges one focused question:
    "Is this a [lower] or [upper] severity violation?"

If judges agree → confirmed severity → auto flag
If judges disagree → human review with both opinions
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SEVERITY_ORDER = ['compliant', 'low', 'medium', 'high', 'critical']

SEVERITY_DESCRIPTIONS = {
    'low':      'Minor mistake. Subtle pressure. Agent may not realise it is wrong. Warning territory.',
    'medium':   'Clear violation. Definite pressure or ignoring hardship. Formal warning territory.',
    'high':     'Serious violation. Abusive, demeaning, or significantly misrepresenting consequences. Notice territory.',
    'critical': 'Severe violation. Explicit threats of legal action, arrest, or contacting third parties. Immediate escalation.'
}


@dataclass
class JudgeSeverityDecision:
    severity: str           # which severity level judge decided
    confidence: float       # how confident (0.0 - 1.0)
    reasoning: str          # why this severity and not the other
    tokens_used: int


@dataclass
class SeverityValidationResult:
    conversation_id: str
    layer1_detected: str            # what Layer 1 said
    sitting_between: Optional[str]  # e.g. "high and critical"
    judge1_severity: str
    judge1_confidence: float
    judge1_reasoning: str
    judge1_tokens: int
    judge2_severity: str
    judge2_confidence: float
    judge2_reasoning: str
    judge2_tokens: int
    judges_agree: bool
    final_severity: str             # confirmed severity or "needs_human_review"
    avg_confidence: float
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            'conversation_id': self.conversation_id,
            'layer1_detected': self.layer1_detected,
            'sitting_between': self.sitting_between,
            'judge1': {
                'severity': self.judge1_severity,
                'confidence': self.judge1_confidence,
                'reasoning': self.judge1_reasoning,
                'tokens': self.judge1_tokens
            },
            'judge2': {
                'severity': self.judge2_severity,
                'confidence': self.judge2_confidence,
                'reasoning': self.judge2_reasoning,
                'tokens': self.judge2_tokens
            },
            'judges_agree': self.judges_agree,
            'final_severity': self.final_severity,
            'avg_confidence': self.avg_confidence,
            'timestamp': self.timestamp
        }


class ConfigLoader:

    @staticmethod
    def load_llm_config(config_file: str = "config/llm_config.json") -> Dict:
        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"LLM config not found: {config_file}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_compliance_rules(rules_file: str = "data/compliance_rules.json") -> Dict:
        with open(rules_file, 'r', encoding='utf-8') as f:
            return json.load(f)


class LLMClient:

    def __init__(self, provider: str, model: str, temperature: float,
                 max_tokens: int, system_prompt: str = None):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or \
            "You are a compliance analyst. Respond only with valid JSON."
        self.client = self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            import openai
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not found")
            return openai.OpenAI(api_key=key)
        elif self.provider == "anthropic":
            import anthropic
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not found")
            return anthropic.Anthropic(api_key=key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def call(self, prompt: str) -> Tuple[str, int]:
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content, response.usage.total_tokens

        elif self.provider == "anthropic":
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": full_prompt}]
            )
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return response.content[0].text, tokens


class SeverityAdjudicator:
    """
    Two LLM judges each answer one focused question:
    'Is this violation [severity_a] or [severity_b]?'

    Judge 1 is strict — defaults to higher severity when uncertain
    Judge 2 is contextual — considers mitigating circumstances

    If they agree → confirmed
    If they disagree → human review
    """

    def __init__(self, config: Dict, rules_data: Dict):
        self.config = config
        self.rules_data = rules_data

        j1 = config['judge1']
        j2 = config['judge2']

        self.judge1 = LLMClient(
            provider=j1['provider'],
            model=j1['model'],
            temperature=j1['temperature'],
            max_tokens=j1['max_tokens'],
            system_prompt=(
                "You are a strict compliance analyst. "
                "When in doubt between two severity levels, choose the higher one. "
                "Respond only with valid JSON."
            )
        )

        self.judge2 = LLMClient(
            provider=j2['provider'],
            model=j2['model'],
            temperature=j2['temperature'],
            max_tokens=j2['max_tokens'],
            system_prompt=(
                "You are a contextual compliance analyst. "
                "Consider the full conversation context and any mitigating circumstances. "
                "Respond only with valid JSON."
            )
        )

        logger.info(f"Judge 1 (Strict):     {j1['model']}")
        logger.info(f"Judge 2 (Contextual): {j2['model']}")

    def _build_prompt(
        self,
        conversation: Dict,
        layer1_result: Dict,
        severity_a: str,
        severity_b: str
    ) -> str:
        """
        Build a focused severity adjudication prompt.
        The question is specific: severity_a or severity_b?
        Not: is this compliant?
        """
        messages_text = "\n".join(
            f"{msg['role'].upper()}: {msg['text']}"
            for msg in conversation['messages']
        )

        evidence_text = "\n".join(layer1_result.get('evidence', []))

        violated_messages = []
        for v in layer1_result.get('violations', []):
            violated_messages.append(
                f"  Message {v['message_index']}: \"{v['matched_text']}\"\n"
                f"  Centroid similarities: {v.get('centroid_similarities', {})}"
            )
        violated_text = "\n".join(violated_messages)

        prompt = f"""A compliance violation has been confirmed in this debt collection call.
Your task is NOT to determine if a violation occurred — that is already confirmed.
Your task is ONLY to determine the severity level.

━━━ CONVERSATION ━━━
{messages_text}

━━━ CONFIRMED VIOLATION ━━━
Layer 1 detected: {layer1_result.get('detected_severity', 'unknown').upper()}
Flagged messages:
{violated_text}

━━━ YOUR QUESTION ━━━
Is this a {severity_a.upper()} or {severity_b.upper()} violation?

━━━ SEVERITY DEFINITIONS ━━━
{severity_a.upper()}: {SEVERITY_DESCRIPTIONS[severity_a]}
{severity_b.upper()}: {SEVERITY_DESCRIPTIONS[severity_b]}

━━━ RULES FOR REFERENCE ━━━
{chr(10).join(f"- {r['id']} ({r['severity'].upper()}): {r['description']}" for r in self.rules_data['rules'])}

Respond with valid JSON only:
{{
  "severity": "{severity_a} or {severity_b}",
  "confidence": 0.0 to 1.0,
  "reasoning": "one paragraph explaining why this severity and not the other"
}}"""

        return prompt

    def _call_judge(
        self,
        judge: LLMClient,
        prompt: str,
        valid_severities: List[str]
    ) -> Optional[JudgeSeverityDecision]:
        try:
            response_text, tokens = judge.call(prompt)

            # Clean JSON
            text = response_text.replace('```json', '').replace('```', '').strip()
            if '{' in text:
                text = text[text.find('{'):text.rfind('}')+1]

            data = json.loads(text)

            severity = data.get('severity', '').lower().strip()

            # Validate it's one of the two options
            if severity not in valid_severities:
                # Try to find a valid severity in the response
                for s in valid_severities:
                    if s in severity:
                        severity = s
                        break
                else:
                    logger.warning(f"Judge returned unexpected severity: {severity}")
                    severity = valid_severities[0]  # default to lower severity

            return JudgeSeverityDecision(
                severity=severity,
                confidence=float(data.get('confidence', 0.5)),
                reasoning=data.get('reasoning', ''),
                tokens_used=tokens
            )

        except Exception as e:
            logger.error(f"Judge error: {str(e)[:100]}")
            return None

    def adjudicate(
        self,
        conversation: Dict,
        layer1_result: Dict
    ) -> SeverityValidationResult:
        """
        Run both judges on the severity question.
        """
        conv_id = layer1_result['conversation_id']
        detected = layer1_result.get('detected_severity', 'medium')
        sitting_between = layer1_result.get('sitting_between')

        # Determine which two severities to adjudicate between
        if sitting_between:
            # Layer 1 told us exactly which boundary
            # e.g. "high and critical" → ['high', 'critical']
            parts = sitting_between.split(' and ')
            severity_a, severity_b = parts[0].strip(), parts[1].strip()
        else:
            # Not on a boundary — adjudicate between detected and one level down
            detected_idx = SEVERITY_ORDER.index(detected)
            lower_idx = max(1, detected_idx - 1)  # don't go below 'low'
            severity_a = SEVERITY_ORDER[lower_idx]
            severity_b = detected

        valid_severities = [severity_a, severity_b]

        logger.info(f"  Adjudicating {conv_id}: {severity_a} vs {severity_b}")

        prompt = self._build_prompt(
            conversation, layer1_result, severity_a, severity_b
        )

        # Judge 1
        j1 = self._call_judge(self.judge1, prompt, valid_severities)
        if j1:
            logger.info(f"    Judge 1 (Strict):     {j1.severity} ({j1.confidence:.2f})")
        time.sleep(0.5)

        # Judge 2
        j2 = self._call_judge(self.judge2, prompt, valid_severities)
        if j2:
            logger.info(f"    Judge 2 (Contextual): {j2.severity} ({j2.confidence:.2f})")

        return self._make_decision(conv_id, detected, sitting_between, j1, j2)

    def _make_decision(
        self,
        conv_id: str,
        detected: str,
        sitting_between: Optional[str],
        j1: Optional[JudgeSeverityDecision],
        j2: Optional[JudgeSeverityDecision]
    ) -> SeverityValidationResult:

        if not j1 or not j2:
            # One judge failed → human review
            return SeverityValidationResult(
                conversation_id=conv_id,
                layer1_detected=detected,
                sitting_between=sitting_between,
                judge1_severity=j1.severity if j1 else 'error',
                judge1_confidence=j1.confidence if j1 else 0.0,
                judge1_reasoning=j1.reasoning if j1 else 'Failed',
                judge1_tokens=j1.tokens_used if j1 else 0,
                judge2_severity=j2.severity if j2 else 'error',
                judge2_confidence=j2.confidence if j2 else 0.0,
                judge2_reasoning=j2.reasoning if j2 else 'Failed',
                judge2_tokens=j2.tokens_used if j2 else 0,
                judges_agree=False,
                final_severity='needs_human_review',
                avg_confidence=0.0,
                timestamp=datetime.now().isoformat()
            )

        judges_agree = j1.severity == j2.severity
        avg_confidence = (j1.confidence + j2.confidence) / 2

        # Both agree and confident → confirmed severity
        # Disagree or low confidence → human review
        agreement_threshold = self.config.get('agreement_threshold', 0.65)

        if judges_agree and avg_confidence >= agreement_threshold:
            final_severity = j1.severity
        else:
            final_severity = 'needs_human_review'

        return SeverityValidationResult(
            conversation_id=conv_id,
            layer1_detected=detected,
            sitting_between=sitting_between,
            judge1_severity=j1.severity,
            judge1_confidence=j1.confidence,
            judge1_reasoning=j1.reasoning,
            judge1_tokens=j1.tokens_used,
            judge2_severity=j2.severity,
            judge2_confidence=j2.confidence,
            judge2_reasoning=j2.reasoning,
            judge2_tokens=j2.tokens_used,
            judges_agree=judges_agree,
            final_severity=final_severity,
            avg_confidence=round(avg_confidence, 3),
            timestamp=datetime.now().isoformat()
        )


class CostTracker:

    COSTS = {
        'gpt-4o-mini':               {'input': 0.150,  'output': 0.600},
        'gpt-4o':                    {'input': 2.50,   'output': 10.00},
        'claude-3-haiku-20240307':   {'input': 0.25,   'output': 1.25},
        'claude-3-5-sonnet-20241022':{'input': 3.00,   'output': 15.00},
    }

    def __init__(self):
        self.total_tokens = 0
        self.models_used = set()
        self.per_conv = []

    def track(self, conv_id: str, result: SeverityValidationResult,
              model1: str, model2: str):
        tokens = result.judge1_tokens + result.judge2_tokens
        self.total_tokens += tokens
        self.models_used.update([model1, model2])
        cost = self._cost(model1, result.judge1_tokens) + \
               self._cost(model2, result.judge2_tokens)
        self.per_conv.append({
            'conversation_id': conv_id,
            'tokens': tokens,
            'cost': round(cost, 5)
        })

    def _cost(self, model: str, tokens: int) -> float:
        if model not in self.COSTS:
            return tokens * 1.0 / 1_000_000
        c = self.COSTS[model]
        return (tokens * 0.6 * c['input'] + tokens * 0.4 * c['output']) / 1_000_000

    def summary(self, total_convs: int) -> Dict:
        total_cost = sum(c['cost'] for c in self.per_conv)
        return {
            'total_tokens': self.total_tokens,
            'total_cost_usd': round(total_cost, 4),
            'cost_per_conversation': round(total_cost / total_convs, 5) if total_convs else 0,
            'per_conversation': self.per_conv
        }


class OutputGenerator:

    def __init__(self, output_dir: str = "data/layer2_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_all(self, results: List[SeverityValidationResult], cost: Dict):
        logger.info("\nGenerating outputs...")

        confirmed = [r for r in results if r.final_severity != 'needs_human_review']
        human = [r for r in results if r.final_severity == 'needs_human_review']

        self._save_json(confirmed, "confirmed_severities.json")
        self._save_json(human, "human_review_needed.json")
        self._save_excel(results, "layer2_results.xlsx")
        self._save_cost(cost)

        logger.info("✅ Outputs generated")

    def _save_json(self, results, filename):
        fp = self.output_dir / filename
        with open(fp, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info(f"  ✓ {fp} ({len(results)} records)")

    def _save_excel(self, results, filename):
        fp = self.output_dir / filename
        rows = [{
            'Conversation ID': r.conversation_id,
            'Layer 1 Detected': r.layer1_detected,
            'Sitting Between': r.sitting_between or '-',
            'Final Severity': r.final_severity,
            'Judges Agree': 'YES' if r.judges_agree else 'NO',
            'Avg Confidence': f"{r.avg_confidence:.2f}",
            'Judge 1 (Strict)': r.judge1_severity,
            'Judge 1 Confidence': f"{r.judge1_confidence:.2f}",
            'Judge 1 Reasoning': r.judge1_reasoning[:200],
            'Judge 2 (Contextual)': r.judge2_severity,
            'Judge 2 Confidence': f"{r.judge2_confidence:.2f}",
            'Judge 2 Reasoning': r.judge2_reasoning[:200],
        } for r in results]

        df = pd.DataFrame(rows)
        with pd.ExcelWriter(fp, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Severity Review', index=False)
            ws = writer.sheets['Severity Review']
            for col in ws.columns:
                col = list(col)
                width = min(max(len(str(c.value or '')) for c in col) + 2, 60)
                ws.column_dimensions[col[0].column_letter].width = width
        logger.info(f"  ✓ {fp} (Excel)")

    def _save_cost(self, cost):
        fp = self.output_dir / "cost_report.json"
        with open(fp, 'w') as f:
            json.dump(cost, f, indent=2)
        logger.info(f"  ✓ {fp}")


def main():
    print("="*70)
    print("LAYER 2: SEVERITY ADJUDICATION")
    print("="*70 + "\n")

    config = ConfigLoader.load_llm_config()
    rules_data = ConfigLoader.load_compliance_rules()

    queue_file = "data/layer1_output/llm_review_queue.json"
    if not Path(queue_file).exists():
        print(f"❌ {queue_file} not found. Run Layer 1 first.")
        return

    with open(queue_file, 'r') as f:
        queue = json.load(f)

    if not queue:
        print("✅ Layer 2 queue is empty — Layer 1 handled everything.")
        return

    with open("data/conversations.json", 'r') as f:
        all_convs = {c['conversation_id']: c for c in json.load(f)}

    logger.info(f"Processing {len(queue)} conversations from Layer 1 queue\n")

    adjudicator = SeverityAdjudicator(config, rules_data)
    cost_tracker = CostTracker()
    results = []

    for i, layer1_item in enumerate(queue, 1):
        conv_id = layer1_item['conversation_id']
        print(f"\n[{i}/{len(queue)}] {conv_id}")
        print(f"  Layer 1 detected: {layer1_item.get('detected_severity', '?').upper()}")
        print(f"  Sitting between:  {layer1_item.get('sitting_between', 'N/A')}")

        conv = all_convs.get(conv_id)
        if not conv:
            logger.error(f"Conversation {conv_id} not found in conversations.json")
            continue

        result = adjudicator.adjudicate(conv, layer1_item)
        results.append(result)

        cost_tracker.track(
            conv_id, result,
            config['judge1']['model'],
            config['judge2']['model']
        )

        print(f"  Final severity:   {result.final_severity.upper()}")
        if not result.judges_agree:
            print(f"  ⚠️  Judges disagreed → human review")

        time.sleep(1.0)

    cost = cost_tracker.summary(len(results))
    OutputGenerator().save_all(results, cost)

    total = len(results)
    confirmed = sum(1 for r in results if r.final_severity != 'needs_human_review')
    human = sum(1 for r in results if r.final_severity == 'needs_human_review')
    agreed = sum(1 for r in results if r.judges_agree)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"\n📊 Results:")
    print(f"  Total adjudicated:  {total}")
    print(f"  Confirmed severity: {confirmed} ({confirmed/total*100:.1f}%)")
    print(f"  Human review:       {human} ({human/total*100:.1f}%)")
    print(f"  Judges agreed:      {agreed} ({agreed/total*100:.1f}%)")
    print(f"\n💰 Cost: ${cost['total_cost_usd']:.4f} "
          f"(${cost['cost_per_conversation']:.5f} per conversation)")
    print(f"\n📁 Outputs in data/layer2_output/\n{'='*70}")


if __name__ == "__main__":
    main()