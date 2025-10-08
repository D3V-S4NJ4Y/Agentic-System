from typing import Any
import sympy
import re
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    @abstractmethod
    def solve(self, problem: str) -> Any:
        pass

class SymbolicSolver(BaseSolver):
    """Handles mathematical expressions using symbolic computation"""
    
    def solve(self, problem: str) -> Any:
        # Extract mathematical expression
        expr = self._extract_expression(problem)
        if not expr:
            raise ValueError("No mathematical expression found")
        
        try:
            # Convert to SymPy expression
            sympy_expr = sympy.sympify(expr)
            
            # Try to solve if it's an equation
            if "=" in expr:
                lhs, rhs = expr.split("=")
                equation = sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))
                return sympy.solve(equation)
            
            # Otherwise evaluate the expression
            return float(sympy_expr.evalf())
        except Exception as e:
            raise ValueError(f"Failed to solve expression: {str(e)}")
    
    def _extract_expression(self, text: str) -> str:
        """Extract mathematical expression from text"""
        # Remove word problems but keep numbers and operators
        text = text.lower()
        
        # Look for explicit equations
        equation_match = re.search(r'\d+\s*[+\-*/=]+\s*\d+', text)
        if equation_match:
            return equation_match.group(0)
        
        # Look for numbers and operations
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 2:
            # Infer operation from keywords
            if any(op in text for op in ["add", "sum", "plus"]):
                return f"{numbers[0]} + {numbers[1]}"
            elif any(op in text for op in ["subtract", "minus", "difference"]):
                return f"{numbers[0]} - {numbers[1]}"
            elif any(op in text for op in ["multiply", "times", "product"]):
                return f"{numbers[0]} * {numbers[1]}"
            elif any(op in text for op in ["divide", "quotient"]):
                return f"{numbers[0]} / {numbers[1]}"
                
        raise ValueError("No mathematical expression found")