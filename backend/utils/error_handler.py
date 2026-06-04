"""
Error Handling & Exception Utilities for Grey Backend

Provides:
- Custom exception classes
- Exception handlers for FastAPI
- Retry logic for transient failures
- Error response normalization
"""

from fastapi import HTTPException, status
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from typing import Any, Dict, Optional
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


class MongoConnectionError(HTTPException):
    """Raised when MongoDB connection fails"""
    def __init__(self, detail: str = "Database connection failed"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail
        )


class ValidationError(HTTPException):
    """Raised when request validation fails"""
    def __init__(self, detail: str = "Invalid request"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )


class NotFoundError(HTTPException):
    """Raised when resource not found"""
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )


def handle_mongo_error(operation_name: str = "operation"):
    """
    Decorator to handle MongoDB errors and return proper HTTP responses
    
    Usage:
        @handle_mongo_error("list campaigns")
        async def list_campaigns():
            mongo = MongoService()
            # ... code ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.error(f"MongoDB connection error during {operation_name}: {str(e)}")
                raise MongoConnectionError(
                    detail=f"Database temporarily unavailable. Please try again in a moment."
                )
            except Exception as e:
                logger.error(f"Unexpected error during {operation_name}: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Internal server error: {type(e).__name__}"
                )
        return wrapper
    return decorator


def retry_on_transient(
    max_attempts: int = 3,
    delay_seconds: float = 0.5,
    backoff_factor: float = 2.0,
    exceptions: tuple = (ConnectionFailure, ServerSelectionTimeoutError)
):
    """
    Retry decorator for transient failures (connection timeouts, etc.)
    
    Usage:
        @retry_on_transient(max_attempts=3)
        def fetch_campaigns():
            mongo = MongoService()
            # ... code ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = delay_seconds
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt} failed, retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator


def normalize_error_response(error: Exception) -> Dict[str, Any]:
    """Convert exception to normalized error response"""
    if isinstance(error, HTTPException):
        return {
            "status": "error",
            "code": error.status_code,
            "message": error.detail
        }
    
    error_type = type(error).__name__
    if error_type in ("ConnectionFailure", "ServerSelectionTimeoutError"):
        return {
            "status": "error",
            "code": 503,
            "message": "Database connection failed",
            "error_type": error_type
        }
    
    return {
        "status": "error",
        "code": 500,
        "message": str(error),
        "error_type": error_type
    }
