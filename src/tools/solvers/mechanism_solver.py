from typing import Dict, List, Any
import numpy as np
from sympy import symbols, solve, Eq
from ..symbolic_solver import SymbolicSolver

class MechanismSolver:
    """Handles mechanical and physical mechanism problems"""
    
    def __init__(self):
        self.symbolic = SymbolicSolver()
    
    def solve(self, problem: Dict[str, Any], steps: List[str]) -> Dict[str, Any]:
        text = problem["problem_statement"].lower()
        
        if self._is_gear_problem(text):
            return self._solve_gear_problem(text, steps)
        elif self._is_pulley_problem(text):
            return self._solve_pulley_problem(text, steps)
        elif self._is_lever_problem(text):
            return self._solve_lever_problem(text, steps)
        else:
            return self._solve_general_mechanism(text, steps)
    
    def _is_gear_problem(self, text: str) -> bool:
        keywords = ["gear", "teeth", "rotate", "revolution", "rpm"]
        return any(word in text for word in keywords)
    
    def _is_pulley_problem(self, text: str) -> bool:
        keywords = ["pulley", "rope", "cable", "lift", "hoist"]
        return any(word in text for word in keywords)
    
    def _is_lever_problem(self, text: str) -> bool:
        keywords = ["lever", "fulcrum", "balance", "seesaw"]
        return any(word in text for word in keywords)
    
    def _solve_gear_problem(self, text: str, steps: List[str]) -> Dict[str, Any]:
        numbers = self.symbolic.extract_numbers(text)
        
        if len(numbers) < 2:
            return {"error": "Insufficient information for gear problem"}
            
        steps.append(f"Found gear parameters: {numbers}")
        
        # Calculate gear ratio and rotational speed
        ratio = numbers[0] / numbers[1]
        steps.append(f"Gear ratio: {ratio}")
        
        if "rpm" in text or "speed" in text:
            if len(numbers) >= 3:
                result = numbers[2] * ratio
                steps.append(f"Calculated output speed: {result} RPM")
                return {"result": result}
                
        return {"result": ratio}
    
    def _solve_pulley_problem(self, text: str, steps: List[str]) -> Dict[str, Any]:
        numbers = self.symbolic.extract_numbers(text)
        
        if not numbers:
            return {"error": "No numerical values found"}
            
        steps.append(f"Found pulley parameters: {numbers}")
        
        # Calculate mechanical advantage
        if "mechanical advantage" in text:
            ma = 2 ** (len(numbers) - 1)  # For compound pulleys
            steps.append(f"Calculated mechanical advantage: {ma}")
            return {"result": ma}
            
        # Calculate force or distance
        force_out = numbers[0] / (2 ** (len(numbers) - 1))
        steps.append(f"Calculated output force: {force_out}")
        return {"result": force_out}
    
    def _solve_lever_problem(self, text: str, steps: List[str]) -> Dict[str, Any]:
        numbers = self.symbolic.extract_numbers(text)
        
        if len(numbers) < 3:
            return {"error": "Insufficient information for lever problem"}
            
        steps.append(f"Found lever parameters: {numbers}")
        
        # Apply lever principle: F1 * d1 = F2 * d2
        if "force" in text:
            force = (numbers[0] * numbers[1]) / numbers[2]
            steps.append(f"Calculated force: {force}")
            return {"result": force}
        else:
            distance = (numbers[0] * numbers[1]) / numbers[2]
            steps.append(f"Calculated distance: {distance}")
            return {"result": distance}
    
    def _solve_general_mechanism(self, text: str, steps: List[str]) -> Dict[str, Any]:
        # Try to solve using basic mechanical principles
        equations = self.symbolic.extract_equations(text)
        if equations:
            steps.append(f"Found equations: {equations}")
            solution = self.symbolic.solve_equations(equations)
            return {"result": solution}
            
        return {"error": "Could not determine mechanism type"}