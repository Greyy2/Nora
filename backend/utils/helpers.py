"""
Common Helper Functions
Shared utilities to avoid code duplication
"""
import math


def safe_float(value, default=0.0):
    """
    Convert value to float, handling None and NaN.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        float or default
    """
    if value is None:
        return default
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """
    Convert value to int, handling None and NaN.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        int or default
    """
    if value is None:
        return default
    try:
        val = int(float(value))
        return val
    except (ValueError, TypeError):
        return default
