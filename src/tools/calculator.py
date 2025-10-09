from typing import Dict, Any, List
import re
import math

class Calculator:
    """Basic calculator for numerical computations"""
    
    def __init__(self):
        self.operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y if y != 0 else float('inf'),
            '**': lambda x, y: x ** y,
            '^': lambda x, y: x ** y
        }
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Perform numerical calculations"""
        try:
            problem_text = problem.get("problem_statement", "")
            
            # Extract numbers
            numbers = self._extract_numbers(problem_text)
            
            # Extract operations
            operations = self._extract_operations(problem_text)
            
            if len(numbers) >= 2 and operations:
                result = self._calculate(numbers, operations)
                return {
                    "solution": result,
                    "method": "numerical_calculation",
                    "numbers": numbers,
                    "operations": operations
                }
            elif len(numbers) == 1:
                # Single number operations (sqrt, factorial, etc.)
                result = self._single_number_operations(numbers[0], problem_text)
                return {
                    "solution": result,
                    "method": "single_number_operation",
                    "number": numbers[0]
                }
                
            return {"solution": None, "error": "Insufficient numerical data"}
            
        except Exception as e:
            return {"solution": None, "error": str(e)}
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(num) for num in numbers]
    
    def _extract_operations(self, text: str) -> List[str]:
        """Extract mathematical operations from text"""
        ops = []
        for op in self.operations.keys():
            if op in text:
                ops.append(op)
        return ops
    
    def _calculate(self, numbers: List[float], operations: List[str]) -> float:
        """Perform calculation with numbers and operations"""
        if not numbers or not operations:
            return 0
            
        result = numbers[0]
        for i, op in enumerate(operations):
            if i + 1 < len(numbers) and op in self.operations:
                result = self.operations[op](result, numbers[i + 1])
                
        return result
    
    def _single_number_operations(self, number: float, text: str) -> float:
        """Perform single number operations"""
        text = text.lower()
        
        if 'sqrt' in text or 'square root' in text:
            return math.sqrt(abs(number))
        elif 'factorial' in text:
            return math.factorial(int(abs(number))) if number >= 0 and number == int(number) else 0
        elif 'square' in text:
            return number ** 2
        elif 'cube' in text:
            return number ** 3
        else:
            return number