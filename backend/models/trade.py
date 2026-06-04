"""
Trade Model - Immutable record of a completed trade

Trade định nghĩa:
- Một giao dịch đã đóng (có entry và exit)
- Event-level immutable record
- Single source of truth cho analytics

Không phải:
- Order (chưa execute)
- Position (đang mở)
- Signal (chưa thành trade)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class TradeDirection(Enum):
    """Trade direction - Type-safe enum"""
    LONG = "long"
    SHORT = "short"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class Trade:
    """
    Một giao dịch hoàn chỉnh (đã đóng)
    
    Immutable record - một khi tạo, không sửa
    
    Attributes:
        entry_time: Thời điểm vào lệnh
        exit_time: Thời điểm thoát lệnh
        direction: LONG hoặc SHORT (enum)
        entry_price: Giá vào
        exit_price: Giá ra
        quantity: Số lượng (float - crypto có precision)
        pnl: Lãi/lỗ NET (đã trừ commission)
        commission: Phí giao dịch (entry + exit)
        pnl_pct: % lãi/lỗ theo notional value (entry_price * quantity)
        signal_time: Thời điểm signal (optional, for attribution)
        exit_reason: Lý do thoát (optional, for debug)
    """
    entry_time: datetime
    exit_time: datetime
    direction: TradeDirection
    entry_price: float
    exit_price: float
    quantity: float  # Float for crypto precision (not int)
    pnl: float  # NET PnL (already includes commission)
    commission: float = 0.0
    pnl_pct: float = 0.0  # % of notional value (entry_price * quantity)
    mfe: float = 0.0  # Max Favorable Excursion (USDT)
    mfe_pct: float = 0.0  # MFE %
    mae: float = 0.0  # Max Adverse Excursion (USDT)
    mae_pct: float = 0.0  # MAE %
    cumulative_pnl: float = 0.0  # Running Total Pne
    signal_time: Optional[datetime] = None
    exit_reason: Optional[str] = ""
    bars: int = 0  # Number of bars duration
    equity_after_exit: float = 0.0  # Equity sau khi exit (for OR tracking)
    
    # OR (On-going Risk) metadata
    or_risk_before: Optional[float] = None  # Risk % before reduction
    or_risk_after: Optional[float] = None   # Risk % after reduction
    or_contracts_before: Optional[float] = None  # Contracts before reduction
    or_contracts_after: Optional[float] = None   # Contracts after reduction
    or_unrealized_pnl: Optional[float] = None  # Unrealized PnL at trigger
    or_on_going_equity: Optional[float] = None  # Equity at trigger (E₀)
    
    # Slippage metadata
    slippage_pct: float = 0.0  # Slippage % used
    entry_price_no_slip: Optional[float] = None  # Entry price without slippage
    exit_price_no_slip: Optional[float] = None   # Exit price without slippage
    slippage_cost: float = 0.0  # Total slippage cost (entry + exit)
    
    def net_pnl(self) -> float:
        """
        Lãi/lỗ sau trừ phí giao dịch
        
        Note: pnl đã bao gồm commission, không cần trừ thêm
        """
        return self.pnl
    
    def is_winner(self) -> bool:
        """Trade này có lãi không? (breakeven không tính win)"""
        return self.pnl > 0
    
    def is_long(self) -> bool:
        """Đây có phải trade LONG không?"""
        return self.direction == TradeDirection.LONG
    
    def is_short(self) -> bool:
        """Đây có phải trade SHORT không?"""
        return self.direction == TradeDirection.SHORT
    
    def holding_time(self) -> float:
        """Thời gian giữ lệnh (giờ)"""
        delta = self.exit_time - self.entry_time
        return delta.total_seconds() / 3600
    
    def to_dict(self) -> dict:
        """
        Convert sang dictionary (cho JSON/MongoDB)
        
        Returns:
            Dict với datetime serialized, direction as string
        """
        return {
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'signal_time': self.signal_time.isoformat() if self.signal_time else None,
            'direction': self.direction.value,  # Enum -> string
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'commission': self.commission,
            'mfe': self.mfe,
            'mfe_pct': self.mfe_pct,
            'mae': self.mae,
            'mae_pct': self.mae_pct,
            'cumulative_pnl': self.cumulative_pnl,
            'exit_reason': self.exit_reason,
            'bars': self.bars,
            'equity_after_exit': self.equity_after_exit,
            'or_risk_before': self.or_risk_before,
            'or_risk_after': self.or_risk_after,
            'or_contracts_before': self.or_contracts_before,
            'or_contracts_after': self.or_contracts_after,
            'or_unrealized_pnl': self.or_unrealized_pnl,
            'or_on_going_equity': self.or_on_going_equity,
            'slippage_pct': self.slippage_pct,
            'entry_price_no_slip': self.entry_price_no_slip,
            'exit_price_no_slip': self.exit_price_no_slip,
            'slippage_cost': self.slippage_cost,
        }
