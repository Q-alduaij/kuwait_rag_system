import re
import logging
from typing import Dict, Any, List, Tuple
from models.schemas import ContentType, SensitivityLevel

logger = logging.getLogger(__name__)

class ContentValidator:
    """Validator for content quality and safety"""
    
    def __init__(self):
        self.min_content_length = 10
        self.max_content_length = 10000
        
        # Sensitive content patterns (would be expanded)
        self.sensitive_patterns = {
            "inappropriate": [],
            "personal_info": [r'\b\d{9,}\b', r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            "offensive_language": []
        }
    
    def validate_content(self, content: str, content_type: ContentType) -> Dict[str, Any]:
        """Validate content for quality and safety"""
        results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "sensitivity_level": self._assess_sensitivity(content, content_type)
        }
        
        # Length validation
        if len(content) < self.min_content_length:
            results["is_valid"] = False
            results["errors"].append(f"Content too short: {len(content)} characters")
        
        if len(content) > self.max_content_length:
            results["warnings"].append(f"Content very long: {len(content)} characters")
        
        # Sensitive content check
        sensitive_issues = self._check_sensitive_content(content)
        if sensitive_issues:
            results["warnings"].extend(sensitive_issues)
            results["sensitivity_level"] = max(results["sensitivity_level"], SensitivityLevel.MEDIUM)
        
        # Content-type specific validation
        type_specific_issues = self._validate_by_content_type(content, content_type)
        if type_specific_issues:
            results["warnings"].extend(type_specific_issues)
        
        return results
    
    def _assess_sensitivity(self, content: str, content_type: ContentType) -> SensitivityLevel:
        """Assess sensitivity level based on content and type"""
        if content_type in [ContentType.QURAN, ContentType.TAFSIR, ContentType.HADITH]:
            return SensitivityLevel.HIGH
        elif content_type == ContentType.LEGAL:
            return SensitivityLevel.MEDIUM
        else:
            return SensitivityLevel.LOW
    
    def _check_sensitive_content(self, content: str) -> List[str]:
        """Check for sensitive or inappropriate content"""
        issues = []
        
        # Check for personal information
        for pattern_type, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    issues.append(f"Potential {pattern_type} detected")
                    break
        
        return issues
    
    def _validate_by_content_type(self, content: str, content_type: ContentType) -> List[str]:
        """Perform content-type specific validation"""
        if content_type == ContentType.QURAN:
            return self._validate_quranic_content(content)
        elif content_type == ContentType.LEGAL:
            return self._validate_legal_content(content)
        else:
            return []
    
    def _validate_quranic_content(self, content: str) -> List[str]:
        """Validate Quranic content"""
        warnings = []
        
        # Check for proper verse formatting
        if "سورة" not in content and "آية" not in content:
            warnings.append("Quranic content missing standard verse indicators")
        
        return warnings
    
    def _validate_legal_content(self, content: str) -> List[str]:
        """Validate legal content"""
        warnings = []
        
        # Check for legal article indicators
        if "المادة" not in content and "مادة" not in content:
            warnings.append("Legal content missing article indicators")
        
        return warnings

# Utility function
def validate_document_content(content: str, filename: str, content_type: ContentType) -> Dict[str, Any]:
    """Convenience function for content validation"""
    validator = ContentValidator()
    return validator.validate_content(content, content_type)