"""Shared logging utilities for sente and Moyo projects."""

import logging
import sys
from typing import Optional


def get_logger(name: str, 
               level: int = logging.INFO,
               format_string: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        if format_string is None:
            format_string = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
    
    return logger


def setup_file_logging(logger: logging.Logger, 
                      file_path: str,
                      level: int = logging.INFO) -> None:
    """Add file logging to an existing logger.
    
    Args:
        logger: Logger to configure
        file_path: Path to log file
        level: Logging level for file handler
    """
    file_handler = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)


def get_quiet_logger(name: str) -> logging.Logger:
    """Get a logger that only shows warnings and errors.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    return get_logger(name, level=logging.WARNING)


def get_verbose_logger(name: str) -> logging.Logger:
    """Get a logger that shows debug information.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    return get_logger(name, level=logging.DEBUG)
