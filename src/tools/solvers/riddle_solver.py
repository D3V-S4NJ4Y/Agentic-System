from typing import Dict, List, Any
import re
from ..symbolic_solver import SymbolicSolver

class RiddleSolver:
    """Handles classic riddles and lateral thinking puzzles"""
    
    def __init__(self):
        self.symbolic = SymbolicSolver()
        # Common riddle patterns and solutions
        self.COMMON_RIDDLES = {
            "bridge_crossing": {
                "pattern": r"bridge.*cross.*time",
                "solution": "Order people to minimize total crossing time, using fastest person to accompany others"
            },
            "water_jug": {
                "pattern": r"jug.*water.*measure",
                "solution": "Pour water between jugs to achieve target volume through intermediate steps"
            },
            "river_crossing": {
                "pattern": r"cross.*river.*(wolf|fox|chicken|goat|sheep)",
                "solution": "Transport items one by one while respecting constraints"
            }
        }
        
        # Lateral thinking patterns and solutions
        self.LATERAL_PATTERNS = {
            "time_based": {
                "pattern": r"clock|time|hour|minute",
                "approach": "Consider time-related aspects and unusual time measurements"
            },
            "perspective": {
                "pattern": r"view|see|look|appear",
                "approach": "Consider different viewpoints and interpretations"
            },
            "state_change": {
                "pattern": r"before|after|change|become",
                "approach": "Look for changes in state or condition"
            }
        }
    
    def solve(self, problem: Dict[str, Any], steps: List[str]) -> Dict[str, Any]:
        text = problem["problem_statement"].lower()
        
        # Check for classic riddle patterns
        for riddle_type, info in self.COMMON_RIDDLES.items():
            if re.search(info["pattern"], text):
                steps.append(f"Identified classic riddle type: {riddle_type}")
                return self._solve_classic_riddle(riddle_type, text, steps)
        
        # Check for lateral thinking patterns
        for pattern_type, info in self.LATERAL_PATTERNS.items():
            if re.search(info["pattern"], text):
                steps.append(f"Identified lateral thinking pattern: {pattern_type}")
                return self._solve_lateral_thinking(pattern_type, text, steps)
        
        # General approach for unknown patterns
        return self._solve_general_riddle(text, steps)
    
    def _solve_classic_riddle(self, riddle_type: str, text: str, steps: List[str]) -> Dict[str, Any]:
        solution_template = self.COMMON_RIDDLES[riddle_type]["solution"]
        
        # Extract relevant numbers and conditions
        numbers = self.symbolic.extract_numbers(text)
        conditions = self._extract_conditions(text)
        
        steps.append(f"Applying solution template: {solution_template}")
        steps.append(f"Found conditions: {conditions}")
        
        # Customize solution based on specific values
        if riddle_type == "bridge_crossing" and numbers:
            return self._solve_bridge_crossing(numbers, conditions)
        elif riddle_type == "water_jug" and numbers:
            return self._solve_water_jug(numbers)
        elif riddle_type == "river_crossing":
            return self._solve_river_crossing(conditions)
            
        return {"solution": solution_template, "explanation": steps}
    
    def _solve_lateral_thinking(self, pattern_type: str, text: str, steps: List[str]) -> Dict[str, Any]:
        approach = self.LATERAL_PATTERNS[pattern_type]["approach"]
        steps.append(f"Using approach: {approach}")
        
        # Apply pattern-specific strategies
        if pattern_type == "time_based":
            return self._analyze_time_aspects(text, steps)
        elif pattern_type == "perspective":
            return self._analyze_perspectives(text, steps)
        elif pattern_type == "state_change":
            return self._analyze_state_changes(text, steps)
            
        return {"approach": approach, "analysis": steps}
    
    def _solve_general_riddle(self, text: str, steps: List[str]) -> Dict[str, Any]:
        # Look for key elements
        key_elements = self._identify_key_elements(text)
        constraints = self._identify_constraints(text)
        
        steps.append(f"Identified key elements: {key_elements}")
        steps.append(f"Identified constraints: {constraints}")
        
        # Try to form logical connections
        connections = self._find_logical_connections(key_elements, constraints)
        
        return {
            "key_elements": key_elements,
            "constraints": constraints,
            "logical_connections": connections,
            "reasoning": steps
        }
    
    def _extract_conditions(self, text: str) -> List[str]:
        # Extract conditional statements
        conditions = re.findall(r'if.*?then.*?[\.|$]', text, re.IGNORECASE)
        return conditions or []
    
    def _identify_key_elements(self, text: str) -> List[str]:
        # Extract nouns and significant terms
        key_terms = re.findall(r'\b[A-Za-z]+(?:\s+[A-Za-z]+)*\b', text)
        return [term for term in key_terms if len(term) > 3]
    
    def _identify_constraints(self, text: str) -> List[str]:
        # Find limiting conditions
        constraints = re.findall(r'must|cannot|only|never|always', text, re.IGNORECASE)
        return constraints
    
    def _find_logical_connections(self, elements: List[str], constraints: List[str]) -> List[str]:
        # Attempt to connect elements based on constraints
        connections = []
        for elem in elements:
            for const in constraints:
                if re.search(rf'\b{elem}\b.*\b{const}\b', ' '.join(elements)):
                    connections.append(f"{elem} is constrained by {const}")
        return connections