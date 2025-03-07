# File: src/safety_utils.py

import re
import logging
import time
from functools import wraps
from typing import Callable, Any

class ContentSafetyFilter:
    """Advanced content safety filtering system."""
    
    INAPPROPRIATE_PATTERNS = [
        # Hate speech patterns
        r'\b(hate|racist|nazi|extremist)\b',
        
        # Violence-related patterns
        r'\b(kill|murder|harm|violent|assault)\b',
        
        # Explicit content patterns
        r'\b(pornographic|sexually explicit|nude)\b',
        
        # Discriminatory language
        r'\b(slur|derogatory|offensive)\b'
    ]
    
    SEVERITY_THRESHOLDS = {
        'low_risk': 1,      # Mild inappropriate content
        'medium_risk': 2,   # Potentially harmful language
        'high_risk': 3      # Severe inappropriate content
    }
    
    @classmethod
    def analyze_content(cls, text: str) -> dict:
        """
        Analyze content for safety risks.
        
        Returns:
            dict with safety assessment
        """
        text = text.lower()
        matches = []
        
        for pattern in cls.INAPPROPRIATE_PATTERNS:
            pattern_matches = re.findall(pattern, text)
            if pattern_matches:
                matches.extend(pattern_matches)
        
        risk_level = 'low_risk'
        if len(matches) >= cls.SEVERITY_THRESHOLDS['high_risk']:
            risk_level = 'high_risk'
        elif len(matches) >= cls.SEVERITY_THRESHOLDS['medium_risk']:
            risk_level = 'medium_risk'
        
        return {
            'is_safe': len(matches) == 0,
            'risk_level': risk_level,
            'matched_patterns': matches
        }
    
    @classmethod
    def filter_content(cls, text: str, replacement: str = '[FILTERED]') -> str:
        """
        Replace inappropriate content with a replacement string.
        """
        for pattern in cls.INAPPROPRIATE_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

class RateLimitHandler:
    """Advanced rate limit handling with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_count = 0
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            while self.retry_count < self.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"Attempt {self.retry_count + 1} failed: {e}")
                    
                    # Exponential backoff
                    delay = self.base_delay * (2 ** self.retry_count)
                    logging.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    
                    self.retry_count += 1
            
            # If all retries fail
            logging.error("Max retries exceeded. Operation failed.")
            raise RuntimeError("Operation failed after multiple attempts")
        
        return wrapper

# Example usage and integration
def safe_gpt_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Apply content safety check before calling
        if 'user_input' in kwargs:
            safety_check = ContentSafetyFilter.analyze_content(kwargs['user_input'])
            
            if not safety_check['is_safe']:
                logging.warning(f"Unsafe content detected: {safety_check['risk_level']}")
                return "Content does not meet safety guidelines."
            
            # Optional: Filter content
            kwargs['user_input'] = ContentSafetyFilter.filter_content(kwargs['user_input'])
        
        # Apply rate limit handling
        rate_limiter = RateLimitHandler()
        return rate_limiter(func)(*args, **kwargs)
    
    return wrapper