"""
Logging setup for Bot Trading V2
"""

import sys
from pathlib import Path
from typing import Optional
import os
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_file_path: Optional[str] = None,
    rotation: Optional[str] = None,
    retention: Optional[str] = None,
):
    """
    Setup logger with custom configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file_path: Path to log file
        rotation: Log rotation interval
        retention: Log retention period
        
    Returns:
        logger: Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )
    
    # Add file handler if enabled
    if log_to_file:
        if log_file_path is None:
            base_path = Path(__file__).parent.parent
            log_file_path = base_path / "logs" / "trading_bot.log"

        # Create logs directory if it doesn't exist
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

        # If rotation is not provided, add a simple non-rotating file sink to avoid
        # Windows file-lock rename PermissionError during short demo runs.
        if rotation:
            logger.add(
                log_file_path,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} - {message}",
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="zip",
            )
        else:
            logger.add(
                log_file_path,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} - {message}",
                level=log_level,
            )
    
    logger.info(f"Logger initialized with level: {log_level}")
    return logger


def get_logger():
    """Get logger instance"""
    return logger


# Initialize logger with default settings
setup_logger()
