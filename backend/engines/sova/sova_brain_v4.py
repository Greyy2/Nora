"""
SOVA LLM: Sovereign Synthesis Engine (v4.0 - Institutional Grade)
Role: The 'Brain' that synthesizes Alpha research into perfect math.
Logic: RAG-Enhanced Thompson Sampling + Grounded Feedback Loop.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from Sova.core.brain_config import settings
from Sova.services.memory_service import memory_bank

logger = logging.getLogger("SOVA.Brain.LLM")

class SovaIntelligence:
    """
    Refactored Sova Brain. 
    Focuses on surgical formula refinement using grounded data.
    """
    
    def __init__(self):
        self.model = settings.PRIMARY_MODEL
        logger.info(f"[SOVA] Brain initialized with {self.model}")

    async def synthesize_alpha(self, 
                               base_idea: str, 
                               diagnostics: List[str], 
                               market_context: str,
                               current_metrics: Dict[str, Any]) -> str:
        """
        The core reasoning loop. 
        Uses RAG to remember what worked and Grounded Diagnostics to fix what's broken.
        """
        
        # 1. Retrieve Past Wins (RAG)
        past_tactics = memory_bank.retrieve_relevant_tactics(market_context)
        tactics_str = "\n".join([f"- {t}" for t in past_tactics])
        
        # 2. Construct Institutional Prompt
        system_prompt = (
            "You are SOVA Sovereign Intelligence. The world's most advanced Quant Researcher. "
            "Your math is flawless. Your logic is grounded in market physics. "
            "You specialize in SURGICAL REFINEMENT of Alpha factors."
        )
        
        user_prompt = f"""
[OBJECTIVE]: Improve IC to > 0.07.
[MARKET CONTEXT]: {market_context}
[CURRENT FORMULA]: {base_idea}
[PERFORMANCE]: IC={current_metrics.get('ic', 0):.4f} | NaN={current_metrics.get('nan_freq', 0):.2%}

[DIAGNOSTIC FEEDBACK]:
{ " | ".join(diagnostics) }

[TACTICAL MEMORY (Successful patterns from similar markets)]:
{tactics_str}

[RULES]:
1. Do not replace the formula, IMPROVE it.
2. Use valid DSL: TS_RANK, CS_ZSCORE, EMA, LOG_RETURN, VWAP_ZSCORE.
3. Fix the specific failures mentioned in Diagnostics.

Return the improved DSL expression ONLY:"""

        # 3. Call LLM (Async)
        try:
            # Note: Assuming sova_chat_completion is the existing global adapter
            from Sova.sova_llm import sova_chat_completion 
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Using asyncio to keep the system responsive
            resp, _ = await asyncio.to_thread(sova_chat_completion, messages)
            refined = self._clean_output(resp)
            
            # 4. Save to Memory if it's a potential win
            if current_metrics.get("ic", 0) > 0.05:
                memory_bank.save_win(refined, current_metrics, market_context)
                
            return refined
        except Exception as e:
            logger.error(f"[SOVA] Synthesis failed: {e}")
            return base_idea

    def _clean_output(self, text: str) -> str:
        if not text: return ""
        text = text.strip().split("\n")[0]
        return text.replace("```", "").strip()

# Global Singleton
sova_brain = SovaIntelligence()
