"""
signal model - Định nghĩa trading signal

signal là gì?
- Một tín hiệu mua/bán tại một thời điểm cụ thể
- Được tạo ra bởi strategy dựa trên indicators
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

class SignalType(Enum):
    """Loại tín hiệu giao dịch"""
    Long = "long" #mua vào
    Short = "short" #bán ra
    Exit_LONG ="exit_long" #đóng lệnh mua
    Exit_SHORT = "exit_short" #đóng lệnh bán khống

@dataclass
class Signal:
    """
    Trading signal tại một thời điểm cụ thể

    Attributes:
        timestamp: Thời điểm signal được tạo ra
        signal_type: Loại signal (LONG, SHORT, EXIT_LONG, EXIT_SHORT)
        price: Gía tại thời điểm signal
        reason: Lý do tạo signal (debug)
    """
    timestamp: datetime
    signal_type: SignalType
    price: float
    reason: str = ""

    def is_entry(self) -> bool:
        """Kiểm tra đây có phải signal vào lệnh không"""
        return self.signal_type in [SignalType.Long, SignalType.Short]

    def is_exit(self) -> bool:
        """Kiểm tra đây có phải signal thoát lệnh không"""
        return self.signal_type in [SignalType.Exit_LONG, SignalType.Exit_SHORT]
    
    def is_long(self) -> bool:
        """Kiểm tra đây có phải signal mua vào không"""
        return self.signal_type == SignalType.Long
    def is_short(self) -> bool:
        """Kiểm tra đây có phải signal bán ra không"""
        return self.signal_type == SignalType.Short