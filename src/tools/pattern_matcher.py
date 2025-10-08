from typing import Any, Dict, List
import re
from .symbolic_solver import BaseSolver

class PatternMatcher(BaseSolver):
    """Pattern matching for logical reasoning"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def solve(self, problem: str) -> Any:
        """
        Solve a logical reasoning problem using pattern matching
        
        Args:
            problem (str): The problem to solve
            
        Returns:
            Any: The solution, typically bool or str
        """
        problem = problem.lower()
        
        # Try each pattern until one matches
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                if match := pattern["regex"].search(problem):
                    return self._apply_pattern(pattern, match, problem)
        
        # If no pattern matches, try generic reasoning
        return self._generic_reasoning(problem)
    
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize regex patterns for different types of logical problems"""
        return {
            "conditional": [
                {
                    "regex": re.compile(r"if\s+(.+?)\s+then\s+(.+?)(?:\.|$)"),
                    "handler": self._handle_conditional
                }
            ],
            "comparison": [
                {
                    "regex": re.compile(r"(?:which|what) is (greater|larger|bigger|more|less|smaller|fewer)"),
                    "handler": self._handle_comparison
                }
            ],
            "sequence": [
                {
                    "regex": re.compile(r"(?:next|following|previous|before) (?:number|term|element)"),
                    "handler": self._handle_sequence
                }
            ]
        }
    
    def _apply_pattern(self, pattern: Dict, match: re.Match, problem: str) -> Any:
        """Apply a matched pattern's handler"""
        try:
            return pattern["handler"](match, problem)
        except Exception as e:
            raise ValueError(f"Failed to apply pattern: {str(e)}")
    
    def _handle_conditional(self, match: re.Match, problem: str) -> bool:
        """Handle if-then logical statements"""
        condition = match.group(1)
        conclusion = match.group(2)
        
        # Check if condition is satisfied in the problem
        condition_met = all(word in problem for word in condition.split())
        
        if condition_met:
            return True
        return False
    
    def _handle_comparison(self, match: re.Match, problem: str) -> str:
        """Handle comparison problems"""
        comparison_type = match.group(1)
        
        # Extract numbers from the problem
        numbers = [float(n) for n in re.findall(r'\d+', problem)]
        
        if not numbers or len(numbers) < 2:
            raise ValueError("Not enough numbers to compare")
        
        if comparison_type in ["greater", "larger", "bigger", "more"]:
            return str(max(numbers))
        else:
            return str(min(numbers))
    
    def _handle_sequence(self, match: re.Match, problem: str) -> Any:
        """Handle sequence problems"""
        # Extract numbers in sequence
        numbers = [int(n) for n in re.findall(r'\d+', problem)]
        
        if len(numbers) < 2:
            raise ValueError("Not enough numbers to determine sequence")
        
        # Try to determine the pattern
        if "next" in problem or "following" in problem:
            diff = numbers[-1] - numbers[-2]
            return numbers[-1] + diff
        else:  # previous/before
            diff = numbers[1] - numbers[0]
            return numbers[0] - diff
    
    def _generic_reasoning(self, problem: str) -> Any:
        """Handle problems that don't match specific patterns"""
        # Look for common logical indicators
        if any(word in problem for word in ["all", "every", "each"]):
            return self._handle_universal_statement(problem)
        elif any(word in problem for word in ["some", "any", "at least"]):
            return self._handle_existential_statement(problem)
        else:
            raise ValueError("Could not determine logical pattern")
    
    def _handle_universal_statement(self, problem: str) -> bool:
        """Handle universal statements (all, every, each)"""
        # Extract the subject and predicate
        words = problem.split()
        try:
            universal_idx = next(i for i, word in enumerate(words) 
                               if word in ["all", "every", "each"])
            subject = words[universal_idx + 1]
            return True  # Simplified - assume statement is true
        except:
            return False
    
    def _handle_existential_statement(self, problem: str) -> bool:
        """Handle existential statements (some, any)"""
        # Extract the subject and predicate
        words = problem.split()
        try:
            existential_idx = next(i for i, word in enumerate(words) 
                                 if word in ["some", "any"])
            subject = words[existential_idx + 1]
            return True  # Simplified - assume statement is true
        except:
            return False