import logging
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import re
from dataclasses import dataclass

@dataclass
class ProblemContext:
    """Stores problem context and intermediate results"""
    problem_text: str
    problem_type: Optional[str] = None
    subtasks: List[str] = None
    tools_used: List[str] = None
    intermediate_results: Dict[str, Any] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []
        if self.tools_used is None:
            self.tools_used = []
        if self.intermediate_results is None:
            self.intermediate_results = {}

class BaseAgent(ABC):
    """Enhanced base agent with improved reasoning capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Problem type detection patterns
        self.type_patterns = {
            'sequence': r'(?:next|pattern|series|sequence|following)',
            'spatial': r'(?:position|distance|arrangement|shape|cube|rotate)',
            'mechanism': r'(?:machine|gear|pulley|lever|wheel|mechanism)',
            'optimization': r'(?:maximum|minimum|optimize|best|efficient)',
            'probability': r'(?:chance|probability|likely|odds)',
            'temporal': r'(?:time|schedule|duration|before|after)',
            'logical': r'(?:if|then|either|or|all|none|some)'
        }
        
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
        
    def _log_step(self, message: str, data: Dict[str, Any] = None):
        """Log a processing step"""
        if data:
            self.logger.info(f"{message}: {data}")
        else:
            self.logger.info(message)
            
    def classify_problem(self, text: str) -> List[str]:
        """Identify the type(s) of problem based on text analysis"""
        text = text.lower()
        problem_types = []
        
        # Log the analysis start
        self._log_step("Starting problem classification")
        
        # Check each pattern
        for p_type, pattern in self.type_patterns.items():
            if re.search(pattern, text):
                problem_types.append(p_type)
                self._log_step(f"Detected problem type: {p_type}")
                
        return problem_types
    
    def extract_numerical_values(self, text: str) -> List[float]:
        """Extract all numerical values from the text"""
        # Match both integers and decimals
        numbers = re.findall(r'-?\d*\.?\d+', text)
        values = [float(n) for n in numbers]
        self._log_step(f"Extracted numerical values", {"values": values})
        return values
    
    def extract_units(self, text: str) -> Dict[str, List[str]]:
        """Extract measurement units from the text"""
        unit_patterns = {
            'length': r'(?:meter|m|km|cm|inch|ft|feet)',
            'time': r'(?:second|minute|hour|day|s|min|hr)',
            'weight': r'(?:gram|kg|pound|lb|g)',
            'speed': r'(?:m/s|mph|km/h)',
            'temperature': r'(?:celsius|fahrenheit|°C|°F)',
            'angle': r'(?:degree|radian|°)'
        }
        
        units = {}
        for u_type, pattern in unit_patterns.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                units[u_type] = matches
                self._log_step(f"Found {u_type} units", {"units": matches})
                
        return units
    
    def verify_solution(self, solution: Dict[str, Any], context: ProblemContext) -> bool:
        """Enhanced verification method"""
        self._log_step("Starting solution verification")
        
        # Basic checks
        if solution.get("solution") is None:
            self._log_step("Verification failed: No solution provided")
            return False
            
        # Check numerical consistency
        if isinstance(solution["solution"], (int, float)):
            numbers = self.extract_numerical_values(context.problem_text)
            if numbers and (min(numbers) > solution["solution"] or solution["solution"] > max(numbers) * 100):
                self._log_step("Verification failed: Solution outside reasonable range")
                return False
                
        # Check step consistency
        steps = solution.get("steps", [])
        if not steps or len(steps) < 2:  # Need at least problem identification and solution
            self._log_step("Verification failed: Insufficient solution steps")
            return False
            
        self._log_step("Solution verification passed")
        return True
    
    def create_solution_template(self) -> Dict[str, Any]:
        """Create a standard solution template"""
        return {
            "solution": None,
            "steps": [],
            "confidence": 0.0,
            "intermediate_results": {},
            "tools_used": [],
            "verification": {
                "passed": False,
                "checks": []
            }
        }