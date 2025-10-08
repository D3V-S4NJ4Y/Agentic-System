from typing import Any
import re
from .symbolic_solver import BaseSolver

class Calculator(BaseSolver):
    """Handles basic numerical calculations"""
    
    def solve(self, problem: str) -> Any:
        numbers = self._extract_numbers(problem)
        operation = self._determine_operation(problem)
        
        if not numbers:
            raise ValueError("No numbers found in the problem")
            
        if not operation:
            return numbers[0] if len(numbers) == 1 else numbers
            
        return self._calculate(numbers, operation)
    
    def _extract_numbers(self, text: str) -> list:
        """Extract all numbers from text"""
        return [float(n) for n in re.findall(r'-?\d*\.?\d+', text)]
    
    def _determine_operation(self, text: str) -> str:
        """Determine the mathematical operation to perform"""
        text = text.lower()
        
        # Map keywords to operations
        operations = {
            "add": "+",
            "plus": "+",
            "sum": "+",
            "total": "+",
            "subtract": "-",
            "minus": "-",
            "difference": "-",
            "multiply": "*",
            "times": "*",
            "product": "*",
            "divide": "/",
            "quotient": "/"
        }
        
        # Check for operation keywords
        for keyword, op in operations.items():
            if keyword in text:
                return op
                
        # Check for mathematical symbols
        symbols = ["+", "-", "*", "/"]
        for symbol in symbols:
            if symbol in text:
                return symbol
                
        return ""
    
    def _calculate(self, numbers: list, operation: str) -> float:
        """Perform the calculation"""
        if len(numbers) < 2:
            return numbers[0]
            
        result = numbers[0]
        for num in numbers[1:]:
            if operation == "+":
                result += num
            elif operation == "-":
                result -= num
            elif operation == "*":
                result *= num
            elif operation == "/" and num != 0:
                result /= num
            else:
                raise ValueError(f"Invalid operation: {operation}")
                
        return result