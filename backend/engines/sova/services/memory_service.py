import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from Sova.core.brain_config import settings

class SovaMemoryBank:
    """
    Long-term Tactical Memory for Sova.
    Stores 'Winning Alpha Seeds' and their market context.
    """
    def __init__(self):
        self.file_path = settings.MEMORY_DIR / "tactical_memory.json"
        self._load_memory()

    def _load_memory(self):
        if self.file_path.exists():
            with open(self.file_path, "r") as f:
                self.data = json.load(f)
        else:
            self.data = {"wins": [], "lessons": []}

    def save_win(self, formula: str, metrics: Dict[str, Any], context: str):
        """Records a successful Alpha for future RAG retrieval."""
        entry = {
            "formula": formula,
            "ic": metrics.get("ic", 0),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.data["wins"].append(entry)
        # Keep only top 100 wins
        self.data["wins"] = sorted(self.data["wins"], key=lambda x: x["ic"], reverse=True)[:100]
        self._persist()

    def retrieve_relevant_tactics(self, current_context: str) -> List[str]:
        """Simple keyword-based retrieval (can be upgraded to full Embedding RAG)."""
        # For now, return top performing formulas from similar contexts
        relevant = [w["formula"] for w in self.data["wins"] if current_context.lower() in w["context"].lower()]
        return relevant[:3]

    def _persist(self):
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=2)

memory_bank = SovaMemoryBank()
