from typing import Dict, Any, List
import re

class PatternMatcher:
    """Pattern matching tool for logical reasoning"""
    
    def __init__(self):
        self.logical_patterns = {
            'if_then': r'if\s+(.+?)\s+then\s+(.+)',
            'either_or': r'either\s+(.+?)\s+or\s+(.+)',
            'all_some_none': r'(all|some|none)\s+(.+)',
            'not': r'not\s+(.+)',
            'and': r'(.+?)\s+and\s+(.+)',
            'or': r'(.+?)\s+or\s+(.+)'
        }
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Match logical patterns and apply reasoning"""
        try:
            problem_text = problem.get("problem_statement", "").lower()
            
            # Find matching patterns
            matches = self._find_patterns(problem_text)
            
            if matches:
                # Apply logical reasoning based on patterns
                reasoning = self._apply_logical_reasoning(matches, problem_text)
                return {
                    "solution": reasoning,
                    "method": "pattern_matching",
                    "patterns_found": list(matches.keys())
                }
            else:
                # Try keyword-based reasoning
                keywords = self._extract_keywords(problem_text)
                reasoning = self._keyword_reasoning(keywords, problem_text)
                return {
                    "solution": reasoning,
                    "method": "keyword_reasoning",
                    "keywords": keywords
                }
                
        except Exception as e:
            return {"solution": None, "error": str(e)}
    
    def _find_patterns(self, text: str) -> Dict[str, List[str]]:
        """Find logical patterns in text"""
        matches = {}
        
        for pattern_name, pattern in self.logical_patterns.items():
            found = re.findall(pattern, text, re.IGNORECASE)
            if found:
                matches[pattern_name] = found
                
        return matches
    
    def _apply_logical_reasoning(self, matches: Dict[str, List[str]], text: str) -> str:
        """Apply logical reasoning based on found patterns"""
        reasoning = []
        
        if 'if_then' in matches:
            for condition, conclusion in matches['if_then']:
                reasoning.append(f"If {condition.strip()}, then {conclusion.strip()}")
                
        if 'either_or' in matches:
            for option1, option2 in matches['either_or']:
                reasoning.append(f"Either {option1.strip()} or {option2.strip()}")
                
        if 'all_some_none' in matches:
            for quantifier, statement in matches['all_some_none']:
                reasoning.append(f"{quantifier.capitalize()} {statement.strip()}")
        
        return ". ".join(reasoning) if reasoning else "Logical pattern identified but reasoning unclear"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        important_words = [
            'true', 'false', 'always', 'never', 'sometimes',
            'possible', 'impossible', 'necessary', 'sufficient',
            'implies', 'because', 'therefore', 'hence', 'thus'
        ]
        
        found_keywords = []
        for word in important_words:
            if word in text:
                found_keywords.append(word)
                
        return found_keywords
    
    def _keyword_reasoning(self, keywords: List[str], text: str) -> str:
        """Apply reasoning based on keywords"""
        if not keywords:
            return "No clear logical structure identified"
            
        if 'true' in keywords and 'false' in keywords:
            return "Statement involves truth values - requires logical evaluation"
        elif 'always' in keywords:
            return "Universal statement - applies to all cases"
        elif 'never' in keywords:
            return "Negative universal statement - applies to no cases"
        elif 'sometimes' in keywords:
            return "Existential statement - applies to some cases"
        else:
            return f"Logical reasoning based on keywords: {', '.join(keywords)}"