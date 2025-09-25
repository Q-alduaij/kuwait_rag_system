import hashlib
import hmac
import os
from typing import Optional
from config.settings import settings

class SecurityUtils:
    """Security-related utilities"""
    
    @staticmethod
    def validate_api_key(provided_key: str, expected_key: str) -> bool:
        """Validate API key using constant-time comparison"""
        return hmac.compare_digest(provided_key, expected_key)
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove directory components
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        dangerous_chars = ['/', '\\', '..', '~']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        return filename

class ContentValidator:
    """Content validation for safety and appropriateness"""
    
    @staticmethod
    def validate_content_safety(text: str, content_type: str) -> Dict[str, bool]:
        """Basic content safety validation"""
        # This would be enhanced with more sophisticated checks
        results = {
            "is_safe": True,
            "has_sensitive_content": False,
            "needs_review": False
        }
        
        # Basic keyword checks (would be expanded)
        sensitive_keywords = []  # Define sensitive terms
        
        for keyword in sensitive_keywords:
            if keyword in text.lower():
                results["has_sensitive_content"] = True
                results["needs_review"] = True
                break
        
        # Length-based checks
        if len(text) < 10:
            results["needs_review"] = True
        
        return results