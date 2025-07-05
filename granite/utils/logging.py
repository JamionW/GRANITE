"""
Logging utilities for GRANITE framework
"""
import logging
import sys
from datetime import datetime
import os


class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to log level
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


def setup_logger(name='GRANITE', level='INFO', log_file=None, use_color=True):
    """
    Setup logger with custom formatting
    
    Parameters:
    -----------
    name : str
        Logger name
    level : str
        Logging level
    log_file : str, optional
        Path to log file
    use_color : bool
        Whether to use colored output
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    if use_color and sys.stdout.isatty():
        formatter = ColoredFormatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggingMixin:
    """Mixin class to add logging capabilities"""
    
    def __init__(self, logger_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(logger_name or self.__class__.__name__)
    
    def log_info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def log_debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def log_progress(self, current, total, message=""):
        """Log progress"""
        percent = (current / total) * 100
        self.logger.info(f"Progress: {current}/{total} ({percent:.1f}%) {message}")


def log_section(logger, title, char='=', width=60):
    """Log a section header"""
    border = char * width
    logger.info(border)
    logger.info(title.center(width))
    logger.info(border)


def log_dict(logger, data_dict, title=None):
    """Log a dictionary in readable format"""
    if title:
        logger.info(f"{title}:")
    
    max_key_len = max(len(str(k)) for k in data_dict.keys())
    
    for key, value in data_dict.items():
        if isinstance(value, float):
            logger.info(f"  {str(key).ljust(max_key_len)}: {value:.4f}")
        else:
            logger.info(f"  {str(key).ljust(max_key_len)}: {value}")


def timed_operation(logger=None, message="Operation"):
    """Decorator to time operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            if logger:
                logger.info(f"Starting: {message}")
            
            result = func(*args, **kwargs)
            
            elapsed = datetime.now() - start_time
            
            if logger:
                logger.info(f"Completed: {message} (took {elapsed})")
            
            return result
        return wrapper
    return decorator