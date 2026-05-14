"""Structured logging for Moyo project."""

import json
import logging
import logging.handlers
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager

from .config.settings import get_settings


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logging."""
    
    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_timestamp
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "message": record.getMessage(),
            "logger": record.name,
        }
        
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.fromtimestamp(record.created).isoformat()
        
        if self.include_level:
            log_entry["level"] = record.levelname
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class MoyoLogger:
    """Enhanced logger with structured logging and context management."""
    
    def __init__(self, name: str, settings=None):
        self.name = name
        self.settings = settings or get_settings()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with configuration."""
        logger = logging.getLogger(self.name)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Set log level
        level = getattr(logging, self.settings.logging.level.upper())
        logger.setLevel(level)
        
        # Create formatter
        if self.settings.logging.structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(self.settings.logging.format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.settings.logging.file_path:
            file_path = Path(self.settings.logging.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=self.settings.logging.max_size_mb * 1024 * 1024,
                backupCount=self.settings.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        return logger
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log with additional context fields."""
        extra_fields = {
            "component": self.name,
            "environment": self.settings.environment,
            **kwargs
        }
        
        # Create a custom log record with extra fields
        record = self.logger.makeRecord(
            self.name, level, "", 0, message, (), None
        )
        record.extra_fields = extra_fields
        
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    @contextmanager
    def operation_context(self, operation: str, **kwargs):
        """Context manager for timing operations."""
        start_time = time.time()
        operation_id = f"{operation}_{int(start_time * 1000)}"
        
        self.info(f"Starting operation: {operation}", 
                 operation=operation, 
                 operation_id=operation_id,
                 **kwargs)
        
        try:
            yield operation_id
            duration = time.time() - start_time
            self.info(f"Completed operation: {operation}", 
                     operation=operation,
                     operation_id=operation_id,
                     duration_seconds=duration,
                     status="success",
                     **kwargs)
        except Exception as e:
            duration = time.time() - start_time
            self.error(f"Failed operation: {operation}", 
                      operation=operation,
                      operation_id=operation_id,
                      duration_seconds=duration,
                      status="error",
                      error=str(e),
                      **kwargs)
            raise


def get_logger(name: str, settings=None) -> MoyoLogger:
    """Get a configured Moyo logger."""
    return MoyoLogger(name, settings)


def setup_logging(settings=None):
    """Setup global logging configuration."""
    settings = settings or get_settings()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.logging.level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    if settings.logging.structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(settings.logging.format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if settings.logging.file_path:
        file_path = Path(settings.logging.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=settings.logging.max_size_mb * 1024 * 1024,
            backupCount=settings.logging.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger
