"""
SOVA SUMMARIZER: Expert Feedback & Causal Reasoning Engine
Role: Analyze experiment results, AST stability, metrics comparison, semantic reasoning
Architecture: Multi-gate validation → Causal analysis → Strategic intent derivation
Intelligence Level: Senior Quant Researcher - Distills raw data into actionable wisdom
"""

import ast
import re
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter, deque
from datetime import datetime

import numpy as np

logger = logging.getLogger("SOVA.Summarizer")


@dataclass
class FeedbackResult:
    decision: bool
    observations: str
    causal_reasoning: str
    suggested_intent: str
    stability_warnings: List[str]
    confidence: float = 0.5
    detailed_gates: Dict[str, Any] = field(default_factory=dict)


COMPLEXITY_PROFILES = {
    "CAPITULATION_CRASH": {"max_depth": 3, "max_operators": 4, "max_nodes": 30, "max_params": 6},
    "EXPONENTIAL_BULL": {"max_depth": 5, "max_operators": 6, "max_nodes": 45, "max_params": 10},
    "CHAOTIC_NOISE": {"max_depth": 2, "max_operators": 3, "max_nodes": 20, "max_params": 4},
    "MEAN_REVERSION_ZONE": {"max_depth": 4, "max_operators": 5, "max_nodes": 35, "max_params": 8},
    "DISTRIBUTION_ANOMALY": {"max_depth": 3, "max_operators": 5, "max_nodes": 35, "max_params": 7},
    "STABLE_ACCUMULATION": {"max_depth": 5, "max_operators": 6, "max_nodes": 45, "max_params": 10},
    "STABLE_EROSION": {"max_depth": 3, "max_operators": 4, "max_nodes": 25, "max_params": 6},
    "NEUTRAL": {"max_depth": 4, "max_operators": 5, "max_nodes": 35, "max_params": 8}
}

IMPROVEMENT_THRESHOLDS = {
    "ic": {"min": 0.01, "improvement_pct": 0.05},
    "sharpe": {"min": 1.8, "improvement_pct": 0.05},
    "rank_ic": {"min": 0.01, "improvement_pct": 0.05},
    "mdd": {"max": 0.30},
    "ann_ret": {"min": 0.0},
}

REDUNDANT_PATTERNS = [
    "RANK(RANK(",
    "ABS(ABS(",
    "SIGN(SIGN(",
    "ZSCORE(ZSCORE(",
    "LOG(LOG(",
    "SQRT(SQRT(",
    "RANK(ABS(RANK(",
]

CAUSAL_TEMPLATES = {
    "EXPLOIT_STRONG": {
        "observations": "High conviction IC ({ic_delta:+.4f}) with robust sharpe. Structure captures {regime} alpha.",
        "causal": "Microstructure edge: {operators} on {window_structure} identifies informed flow. SNR is optimized.",
        "intent": "MINER: Fine-tune parameters. Lock core structure; optimize windows for peak IR."
    },
    "EXPLOIT_MODERATE": {
        "observations": "Plausible signal ({ic_delta:+.4f} IC). Structure is directionally correct but lacks amplitude.",
        "causal": "Logical bridge: {operators} partially decodes {regime} dynamics. Secondary noise persists.",
        "intent": "MINER: Amplify signal. Add one interaction layer or switch primary variable to $volume."
    },
    "SIMPLIFY_OVERFIT": {
        "observations": "Structural bloat. Complexity ({depth} depth) fit historical noise, not alpha.",
        "causal": "Over-parameterization: {num_ops} ops dilute the core thesis. Logic is non-generalizable.",
        "intent": "MINER: Strip to 2-op core. Remove redundant RANK/ABS layers. Restore interpretability."
    },
    "SIMPLIFY_UNSTABLE": {
        "observations": "Structural fragility. {num_warnings} stability failures detected in {regime} check.",
        "causal": "Numerical risk: {warning_summary}. Pattern likely to collapse during regime shifts.",
        "intent": "MINER: Stabilize via smoothing (EMA) or normalization. Fix division safety."
    },
    "DIVERSIFY_STAGNANT": {
        "observations": "Zero delta ({ic_delta:+.4f}). This branch is a local optimum of noise.",
        "causal": "Synaptic redundancy: Expression mirrors existing shards. No new information captured.",
        "intent": "QUANTA: Abandon limb. Spawn new branch with different operator family or cross-asset logic."
    },
    "DIVERSIFY_REDUNDANT": {
        "observations": "Semantic overlap. Formula is a minor variant of stored state.",
        "causal": "Convergence trap: Mutation is circular. Search space is saturated at this limb.",
        "intent": "QUANTA: High-mutation jump. Switch foundation from {regime} to VOLATILITY or SPECTRAL."
    },
    "REJECT_DEGRADED": {
        "observations": "Significant decay. Sharpe drop ({sharpe_delta:.2f}) and MDD expansion.",
        "causal": "Signal inversion: {primary_failure}. Complexity added noise without predictive value.",
        "intent": "QUANTA: Hard reset. Pivot to a different Quanta limb immediately."
    },
    "REJECT_DANGEROUS": {
        "observations": "Risk failure: {failure_type}. Dangerous drawdown profile.",
        "causal": "Structural hazard: Formula introduces systemic risk. Incompatible with risk-inertia gates.",
        "intent": "QUANTA: Blacklist limb. Restart from different Golden Seed foundation."
    }
}


class StabilityAnalyzer:
    def __init__(self):
        self.redundant_patterns = REDUNDANT_PATTERNS

    def analyze(self, expression: str, regime: str = "NEUTRAL") -> Tuple[List[str], Dict[str, Any]]:
        warnings = []
        gate_results = {}
        
        profile = COMPLEXITY_PROFILES.get(regime, COMPLEXITY_PROFILES["NEUTRAL"])
        
        depth = expression.count("(")
        gate_results["depth"] = {"value": depth, "max": profile["max_depth"], "passed": depth <= profile["max_depth"]}
        if depth > profile["max_depth"]:
            warnings.append(f"Depth Gate [{regime}]: Nesting depth ({depth}) exceeds regime limit ({profile['max_depth']}). Over-complex for current market conditions.")
        
        operators = re.findall(r'[A-Z_]+\(', expression)
        num_ops = len(operators)
        gate_results["operators"] = {"count": num_ops, "max": profile["max_operators"], "passed": num_ops <= profile["max_operators"]}
        if num_ops > profile["max_operators"]:
            warnings.append(f"Operator Gate [{regime}]: {num_ops} operators exceeds regime limit ({profile['max_operators']}). Reduce computational complexity.")
        
        for pattern in self.redundant_patterns:
            if pattern in expression.upper():
                warnings.append(f"Redundancy Gate: '{pattern.rstrip('(')}' detected. Nested identical operations add computational cost without signal improvement.")
                gate_results["redundancy"] = {"passed": False, "pattern": pattern}
                break
        else:
            gate_results["redundancy"] = {"passed": True}
        
        try:
            clean = expression.replace("$", "VAR_")
            wrapped = f"alpha = {clean}"
            tree = ast.parse(wrapped)
            nodes = list(ast.walk(tree))
            gate_results["ast_nodes"] = {"count": len(nodes), "max": profile["max_nodes"], "passed": len(nodes) <= profile["max_nodes"]}
            if len(nodes) > profile["max_nodes"]:
                warnings.append(f"AST Complexity Gate [{regime}]: {len(nodes)} AST nodes exceeds limit ({profile['max_nodes']}). Risk of overfitting to noise.")
        except SyntaxError as e:
            warnings.append(f"Syntax Gate: Invalid expression structure. Parse error: {str(e)[:100]}")
            gate_results["ast_nodes"] = {"error": str(e), "passed": False}
        except Exception:
            gate_results["ast_nodes"] = {"passed": True}
        
        params = re.findall(r'(?:,\s*)\d+', expression)
        num_params = len(params)
        gate_results["parameters"] = {"count": num_params, "max": profile["max_params"], "passed": num_params <= profile["max_params"]}
        if num_params > profile["max_params"]:
            warnings.append(f"Parameter Gate [{regime}]: {num_params} parameters exceeds limit ({profile['max_params']}). Each parameter increases degrees of freedom for overfitting.")
        
        unique_ops = set(op.rstrip('(') for op in operators)
        op_diversity = len(unique_ops) / (num_ops + 1e-8)
        gate_results["diversity"] = {"unique": len(unique_ops), "total": num_ops, "ratio": op_diversity}
        if num_ops > 3 and op_diversity < 0.4:
            warnings.append(f"Diversity Gate: Only {len(unique_ops)} unique operators out of {num_ops} total. Low diversity suggests structural redundancy.")
        
        return warnings, gate_results


class MetricsComparator:
    def __init__(self, thresholds: Dict[str, Dict] = None):
        self.thresholds = thresholds or IMPROVEMENT_THRESHOLDS

    def compare(self, current: Dict[str, float], sota: Optional[Dict[str, float]]) -> Tuple[bool, Dict[str, Any]]:
        comparison = {}
        
        current_ic = current.get("ic", 0)
        current_sharpe = current.get("sharpe", 0)
        current_mdd = current.get("mdd", 0)
        current_rank_ic = current.get("rank_ic", 0)
        current_ann_ret = current.get("ann_ret", 0)
        
        comparison["ic"] = {"current": current_ic}
        comparison["sharpe"] = {"current": current_sharpe}
        comparison["mdd"] = {"current": current_mdd}
        comparison["rank_ic"] = {"current": current_rank_ic}
        
        if current_ic < self.thresholds["ic"]["min"]:
            comparison["ic"]["gate"] = "BELOW_MIN"
        if current_sharpe < self.thresholds["sharpe"]["min"]:
            comparison["sharpe"]["gate"] = "BELOW_MIN"
        if current_mdd > self.thresholds["mdd"]["max"]:
            comparison["mdd"]["gate"] = "EXCEEDS_MAX"
        
        is_valid = (
            current_ic >= self.thresholds["ic"]["min"]
            and current_sharpe >= self.thresholds["sharpe"]["min"]
            and current_mdd <= self.thresholds["mdd"]["max"]
        )
        
        if not sota:
            return is_valid, comparison
        
        sota_ic = sota.get("ic", 0)
        sota_sharpe = sota.get("sharpe", 0)
        sota_mdd = sota.get("mdd", 1.0)
        
        ic_delta = current_ic - sota_ic
        sharpe_delta = current_sharpe - sota_sharpe
        mdd_delta = current_mdd - sota_mdd
        
        comparison["ic"]["delta"] = ic_delta
        comparison["sharpe"]["delta"] = sharpe_delta
        comparison["mdd"]["delta"] = mdd_delta
        
        ic_improved = current_ic > sota_ic * (1 + self.thresholds["ic"]["improvement_pct"])
        sharpe_improved = current_sharpe > sota_sharpe * (1 + self.thresholds["sharpe"]["improvement_pct"])
        mdd_improved = current_mdd <= sota_mdd
        
        comparison["ic"]["improved"] = ic_improved
        comparison["sharpe"]["improved"] = sharpe_improved
        comparison["mdd"]["improved"] = mdd_improved
        
        is_improvement = is_valid and (ic_improved or sharpe_improved) and mdd_improved
        
        return is_improvement, comparison


class SemanticReasoner:
    def __init__(self):
        self.templates = CAUSAL_TEMPLATES
        self.reasoning_history: deque = deque(maxlen=100)

    def reason(self, experiment: Dict[str, Any], is_improvement: bool,
              stability_warnings: List[str], metric_comparison: Dict[str, Any],
              gate_results: Dict[str, Any]) -> Dict[str, str]:
        metrics = experiment.get("metrics", {})
        expression = experiment.get("expression", "")
        regime = experiment.get("regime", "NEUTRAL")
        
        operators = re.findall(r'[A-Z_]+(?=\()', expression)
        windows = re.findall(r'(?:,\s*)\d+', expression)
        
        context = {
            "regime": regime,
            "operators": ", ".join(set(operators)) if operators else "basic",
            "window_structure": f"windows=[{', '.join(w.strip(', ') for w in windows)}]" if windows else "default windows",
            "ic_delta": metric_comparison.get("ic", {}).get("delta", 0),
            "sharpe_delta": metric_comparison.get("sharpe", {}).get("delta", 0),
            "mdd": metrics.get("mdd", 0),
            "depth": gate_results.get("depth", {}).get("value", 0),
            "num_ops": gate_results.get("operators", {}).get("count", 0),
            "num_warnings": len(stability_warnings),
            "warning_summary": "; ".join(stability_warnings[:2]) if stability_warnings else "none",
        }
        
        template_key = self._select_template(is_improvement, stability_warnings, metric_comparison, context)
        template = self.templates.get(template_key, self.templates["DIVERSIFY_STAGNANT"])
        
        result = {}
        for key in ("observations", "causal", "intent"):
            try:
                text = template[key].format(**context)
                # Axis 13: Enforce strict brevity (<100 words per section)
                words = text.split()
                if len(words) > 80:
                    text = " ".join(words[:75]) + "... [Brevity Enforced]"
                result[key] = text
            except KeyError:
                result[key] = template[key]
        
        ic_current = metrics.get("ic", 0)
        sharpe_current = metrics.get("sharpe", 0)
        
        if is_improvement:
            ic_delta = context.get("ic_delta", 0)
            if ic_delta > 0.01:
                context["primary_improvement"] = f"IC gain of {ic_delta:+.4f}"
            else:
                context["primary_improvement"] = f"Sharpe improvement of {context.get('sharpe_delta', 0):+.2f}"
        
        self.reasoning_history.append({
            "template": template_key,
            "regime": regime,
            "decision": is_improvement and not stability_warnings,
            "intent": result.get("intent", ""),
            "timestamp": datetime.now().isoformat()
        })
        
        return result

    def _select_template(self, is_improvement: bool, warnings: List[str],
                        comparison: Dict[str, Any], context: Dict) -> str:
        ic_delta = context.get("ic_delta", 0)
        sharpe_delta = context.get("sharpe_delta", 0)
        mdd = context.get("mdd", 0)
        
        if mdd > 0.35 or sharpe_delta < -1.0:
            context["failure_type"] = f"Extreme drawdown ({mdd:.1%}) or Sharpe collapse ({sharpe_delta:+.2f})"
            return "REJECT_DANGEROUS"
        
        if ic_delta < -0.02 and sharpe_delta < -0.5:
            context["primary_failure"] = "Reversed signal polarity or noise amplification"
            return "REJECT_DEGRADED"
        
        if warnings:
            depth = context.get("depth", 0)
            num_ops = context.get("num_ops", 0)
            if any("Redundancy" in w for w in warnings):
                return "SIMPLIFY_OVERFIT"
            elif depth > 5 or num_ops > 6:
                return "SIMPLIFY_OVERFIT"
            else:
                return "SIMPLIFY_UNSTABLE"
        
        if is_improvement:
            if ic_delta > 0.005 or sharpe_delta > 0.3:
                return "EXPLOIT_STRONG"
            else:
                return "EXPLOIT_MODERATE"
        
        if abs(ic_delta) < 0.002:
            return "DIVERSIFY_STAGNANT"
        
        return "DIVERSIFY_REDUNDANT"


class StrategicSummarizer:
    def __init__(self, settings: Dict[str, Any] = None):
        self.settings = settings or {}
        self.stability = StabilityAnalyzer()
        self.comparator = MetricsComparator()
        self.reasoner = SemanticReasoner()
        self.feedback_history: deque = deque(maxlen=200)

    def generate_feedback(self, experiment: Dict[str, Any],
                         sota_metrics: Optional[Dict[str, float]] = None) -> FeedbackResult:
        expression = experiment.get("expression", "")
        current_metrics = experiment.get("metrics", {})
        regime = experiment.get("regime", "NEUTRAL")
        
        stability_warnings, gate_results = self.stability.analyze(expression, regime)
        
        is_improvement, metric_comparison = self.comparator.compare(current_metrics, sota_metrics)
        
        reasoning = self.reasoner.reason(experiment, is_improvement, stability_warnings, metric_comparison, gate_results)
        
        decision = is_improvement and not stability_warnings
        
        confidence = self._compute_confidence(is_improvement, stability_warnings, metric_comparison, gate_results)
        
        result = FeedbackResult(
            decision=decision,
            observations=reasoning.get("observations", ""),
            causal_reasoning=reasoning.get("causal", ""),
            suggested_intent=reasoning.get("intent", ""),
            stability_warnings=stability_warnings,
            confidence=confidence,
            detailed_gates={
                "stability": gate_results,
                "metrics": metric_comparison,
                "decision": decision,
                "regime": regime
            }
        )
        
        self.feedback_history.append({
            "expression": expression[:80],
            "regime": regime,
            "decision": decision,
            "intent": result.suggested_intent,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        return result

    def _compute_confidence(self, is_improvement: bool, warnings: List[str],
                           comparison: Dict[str, Any], gate_results: Dict[str, Any]) -> float:
        confidence = 0.5
        
        if is_improvement:
            confidence += 0.2
            ic_delta = comparison.get("ic", {}).get("delta", 0)
            if ic_delta > 0.01:
                confidence += 0.1
            if ic_delta > 0.02:
                confidence += 0.1
        else:
            confidence -= 0.1
        
        if warnings:
            confidence -= 0.1 * min(len(warnings), 3)
        
        gates_passed = sum(1 for g in gate_results.values() if isinstance(g, dict) and g.get("passed", True))
        total_gates = sum(1 for g in gate_results.values() if isinstance(g, dict) and "passed" in g)
        if total_gates > 0:
            gate_ratio = gates_passed / total_gates
            confidence *= (0.5 + 0.5 * gate_ratio)
        
        return max(0.05, min(0.95, confidence))

    def get_history_summary(self) -> Dict[str, Any]:
        if not self.feedback_history:
            return {"message": "No feedback history"}
        
        decisions = [h["decision"] for h in self.feedback_history]
        intents = Counter(h["intent"].split(":")[0] if ":" in h["intent"] else h["intent"]
                         for h in self.feedback_history)
        
        return {
            "total_feedbacks": len(self.feedback_history),
            "approval_rate": sum(decisions) / len(decisions),
            "intent_distribution": dict(intents),
            "avg_confidence": np.mean([h["confidence"] for h in self.feedback_history]),
            "recent": list(self.feedback_history)[-5:]
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    
    logger.info("="*60)
    logger.info("SOVA SUMMARIZER - Standalone Test")
    logger.info("="*60)
    
    summarizer = StrategicSummarizer()
    
    expr = "RANK(DELTA($close, 5))"
    experiment = {
        "expression": expr,
        "metrics": {"ic": 0.045, "sharpe": 2.5, "mdd": 0.08, "rank_ic": 0.04, "ann_ret": 0.35},
        "regime": "STABLE_ACCUMULATION"
    }
    sota = {"ic": 0.040, "sharpe": 2.3, "mdd": 0.10}
    
    feedback = summarizer.generate_feedback(experiment, sota)
    logger.info(f"\nTest 1: Simple improvement")
    logger.info(f"  Decision: {feedback.decision}")
    logger.info(f"  Confidence: {feedback.confidence:.2f}")
    logger.info(f"  Intent: {feedback.suggested_intent}")
    logger.info(f"  Observations: {feedback.observations}")
    logger.info(f"  Causal: {feedback.causal_reasoning}")
    logger.info(f"  Warnings: {feedback.stability_warnings}")
    
    complex_expr = "RANK(RANK(ZSCORE(TS_CORR(DELTA(TS_MEAN($close, 5), 3), $volume, 20), 10)))"
    experiment2 = {
        "expression": complex_expr,
        "metrics": {"ic": 0.050, "sharpe": 2.6, "mdd": 0.09},
        "regime": "CHAOTIC_NOISE"
    }
    
    feedback2 = summarizer.generate_feedback(experiment2, sota)
    logger.info(f"\nTest 2: Complex in chaotic regime")
    logger.info(f"  Decision: {feedback2.decision}")
    logger.info(f"  Confidence: {feedback2.confidence:.2f}")
    logger.info(f"  Intent: {feedback2.suggested_intent}")
    logger.info(f"  Warnings: {feedback2.stability_warnings}")
    
    experiment3 = {
        "expression": "RANK(DELTA($close, 5))",
        "metrics": {"ic": 0.005, "sharpe": 1.2, "mdd": 0.25},
        "regime": "NEUTRAL"
    }
    
    feedback3 = summarizer.generate_feedback(experiment3, sota)
    logger.info(f"\nTest 3: Underperforming")
    logger.info(f"  Decision: {feedback3.decision}")
    logger.info(f"  Confidence: {feedback3.confidence:.2f}")
    logger.info(f"  Intent: {feedback3.suggested_intent}")
    
    summary = summarizer.get_history_summary()
    logger.info(f"\nHistory: {json.dumps(summary, indent=2, default=str)}")
    
    logger.info("="*60)
    logger.info("SOVA SUMMARIZER Test Complete")
    logger.info("="*60)
