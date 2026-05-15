# src/logger.py
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

# Default log directory and file naming
LOG_DIR = "logs"
MAX_LOG_SIZE_MB = 5  # Max size of each log file in MB
BACKUP_COUNT = 5      # Number of backup log files to keep


def setup_logger(
    name: str = "student_performance",
    log_level: Optional[str] = None,
    log_dir: str = LOG_DIR,
) -> logging.Logger:
    """
    Setup and return a configured logger with both file and console handlers.

    Args:
        name: Name of the logger (typically __name__ of the calling module).
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). 
                   If None, reads from env var LOG_LEVEL or defaults to INFO.
        log_dir: Directory to store log files.

    Returns:
        Configured logging.Logger instance.
    """
    # Determine log level
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # --- File handler with rotation ---
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,  # Convert MB to bytes
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    # --- Console handler (for development) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # --- Formatter ---
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Create a default logger instance for general use
logger = setup_logger()
