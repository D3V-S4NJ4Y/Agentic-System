from typing import Dict, List, Any
import numpy as np
from ..symbolic_solver import SymbolicSolver

class SequenceSolver:
    """Handles sequence and pattern-based problems"""
    
    def __init__(self):
        self.symbolic = SymbolicSolver()
        self.sequence_types = {
            "arithmetic": self._check_arithmetic,
            "geometric": self._check_geometric,
            "fibonacci": self._check_fibonacci,
            "polynomial": self._check_polynomial
        }
    
    def solve(self, problem: Dict[str, Any], steps: List[str]) -> Dict[str, Any]:
        text = problem["problem_statement"].lower()
        numbers = self.symbolic.extract_numbers(text)
        
        if not numbers:
            return {"error": "No sequence found in problem"}
            
        steps.append(f"Found sequence: {numbers}")
        
        # Try each sequence type
        for seq_type, checker in self.sequence_types.items():
            if checker(numbers):
                steps.append(f"Identified {seq_type} sequence")
                next_term = self._get_next_term(numbers, seq_type)
                return {"result": next_term, "sequence_type": seq_type}
        
        # If no pattern found, try polynomial fit
        steps.append("No standard sequence pattern found, trying polynomial fit")
        next_term = self._polynomial_prediction(numbers)
        return {"result": next_term, "sequence_type": "polynomial"}
    
    def _check_arithmetic(self, sequence: List[float]) -> bool:
        """Check if sequence is arithmetic (constant difference)"""
        if len(sequence) < 3:
            return False
            
        differences = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
        return all(abs(d - differences[0]) < 1e-10 for d in differences)
    
    def _check_geometric(self, sequence: List[float]) -> bool:
        """Check if sequence is geometric (constant ratio)"""
        if len(sequence) < 3 or 0 in sequence:
            return False
            
        ratios = [sequence[i+1]/sequence[i] for i in range(len(sequence)-1)]
        return all(abs(r - ratios[0]) < 1e-10 for r in ratios)
    
    def _check_fibonacci(self, sequence: List[float]) -> bool:
        """Check if sequence follows Fibonacci pattern"""
        if len(sequence) < 3:
            return False
            
        return all(abs(sequence[i+2] - (sequence[i+1] + sequence[i])) < 1e-10 
                  for i in range(len(sequence)-2))
    
    def _check_polynomial(self, sequence: List[float]) -> bool:
        """Check if sequence follows a polynomial pattern"""
        if len(sequence) < 4:
            return False
            
        # Try fitting polynomials of different degrees
        x = np.arange(len(sequence))
        for degree in range(1, min(4, len(sequence)-1)):
            coeffs = np.polyfit(x, sequence, degree)
            predicted = np.polyval(coeffs, x)
            if np.allclose(predicted, sequence, rtol=1e-10):
                return True
        return False
    
    def _get_next_term(self, sequence: List[float], seq_type: str) -> float:
        """Generate next term based on sequence type"""
        if seq_type == "arithmetic":
            diff = sequence[1] - sequence[0]
            return sequence[-1] + diff
            
        elif seq_type == "geometric":
            ratio = sequence[1] / sequence[0]
            return sequence[-1] * ratio
            
        elif seq_type == "fibonacci":
            return sequence[-1] + sequence[-2]
            
        else:  # polynomial or unknown
            return self._polynomial_prediction(sequence)
    
    def _polynomial_prediction(self, sequence: List[float]) -> float:
        """Predict next term using polynomial regression"""
        x = np.arange(len(sequence))
        # Try polynomials up to degree 3
        best_degree = min(3, len(sequence)-1)
        coeffs = np.polyfit(x, sequence, best_degree)
        return np.polyval(coeffs, len(sequence))