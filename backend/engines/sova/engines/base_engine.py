#!/usr/bin/env python3
"""
Base Alpha Engine Interface
Defines the contract that Stock and Forex engines must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd


class AlphaEngine(ABC):
    """Base interface for market-specific Alpha generation and evaluation."""
    
    @abstractmethod
    def get_axioms(self) -> Dict[str, List[str]]:
        """Return market-specific reasoning axioms."""
        pass
    
    @abstractmethod
    def get_gene_registry(self) -> Dict[str, List[str]]:
        """Return market-specific operator/variable registry."""
        pass
    
    @abstractmethod
    def get_strategy_map(self) -> Dict[str, Dict[str, Any]]:
        """Return regime→strategy mapping."""
        pass
    
    @abstractmethod
    def wrap_expression(self, expr: str) -> str:
        """Apply market-specific normalization wrapping."""
        pass
    
    @abstractmethod
    def zscore_expression(self, expr: str, window: int) -> str:
        """Generate a z-score expression."""
        pass
    
    @abstractmethod
    def get_mutation_wraps(self) -> List[str]:
        """Return valid mutation wrap operators."""
        pass
    
    @abstractmethod
    def mutate_wrap(self, expression: str, wrap_op: str) -> str:
        """Apply a specific mutation wrap."""
        pass
    
    @abstractmethod
    def run_backtest(self, factor: pd.Series, df: pd.DataFrame, expression: str) -> Any:
        """Run market-specific backtesting."""
        pass
    
    @abstractmethod
    def load_data(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Load market-specific data."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name for logging."""
        pass
