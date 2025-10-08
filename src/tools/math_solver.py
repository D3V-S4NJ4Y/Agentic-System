from typing import List, Dict, Any
import re
import numpy as np
from sympy import symbols, solve, Eq, simplify
from .symbolic_solver import BaseSolver

class MathSolver(BaseSolver):
    """Advanced mathematical problem solver with pattern recognition"""
    
    def __init__(self):
        self.name = "MathSolver"
        
    def solve(self, problem: str) -> Any:
        """Main entry point for solving mathematical problems"""
        steps = []
        try:
            # Process the problem
            text = problem.lower() if isinstance(problem, str) else str(problem)
            numbers = self._extract_numbers(text)
            expressions = self._extract_expressions(text)
            
            # Choose solution strategy
            if self._is_sequence_problem(text):
                return self._solve_sequence(text, steps)
            elif self._is_algebra_problem(text):
                return self._solve_algebra(text, steps)
            elif self._is_geometry_problem(text):
                return self._solve_geometry(text, steps)
            else:
                return self._solve_general_math(text, numbers, expressions, steps)
                
        except Exception as e:
            steps.append(f"Error: {str(e)}")
            return None
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text"""
        return [float(n) for n in re.findall(r'-?\d*\.?\d+', text)]
    
    def _extract_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions"""
        return re.findall(r'[-+]?\d*\.?\d+\s*[-+*/]\s*\d*\.?\d+', text)
    
    def _is_sequence_problem(self, text: str) -> bool:
        """Determine if this is a sequence problem"""
        sequence_keywords = [
            "sequence", "pattern", "series", "next number",
            "following", "continues", "progression"
        ]
        return any(keyword in text.lower() for keyword in sequence_keywords)
    
    def _is_algebra_problem(self, text: str) -> bool:
        """Determine if this is an algebraic problem"""
        algebra_keywords = [
            "equation", "solve for", "variable", "unknown",
            "algebra", "expression", "simplify", "factor"
        ]
        return any(keyword in text.lower() for keyword in algebra_keywords)
    
    def _is_geometry_problem(self, text: str) -> bool:
        """Determine if this is a geometry problem"""
        geometry_keywords = [
            "triangle", "square", "circle", "rectangle",
            "area", "perimeter", "volume", "angle",
            "diameter", "radius", "polygon", "shape"
        ]
        return any(keyword in text.lower() for keyword in geometry_keywords)
    
    def _solve_sequence(self, text: str, steps: List[str]) -> Any:
        """Solve sequence problems"""
        # Extract sequence numbers
        sequence_match = re.search(r'(\d+(?:,\s*\d+)*)', text)
        if not sequence_match:
            steps.append("No numeric sequence found in the problem")
            return None
            
        # Parse sequence
        sequence_str = sequence_match.group(1)
        sequence = [int(n.strip()) for n in sequence_str.split(',')]
        steps.append(f"Found sequence: {sequence}")
        
        # Try different sequence patterns
        if len(sequence) >= 3:
            # Arithmetic sequence
            diff = sequence[1] - sequence[0]
            if all(sequence[i+1] - sequence[i] == diff for i in range(len(sequence)-1)):
                steps.append(f"Identified arithmetic sequence with difference {diff}")
                next_num = sequence[-1] + diff
                steps.append(f"Next number in sequence: {next_num}")
                return next_num
                
            # Geometric sequence
            if sequence[0] != 0:
                ratio = sequence[1] / sequence[0]
                if all(abs(sequence[i+1] / sequence[i] - ratio) < 1e-10 for i in range(len(sequence)-1)):
                    steps.append(f"Identified geometric sequence with ratio {ratio}")
                    next_num = sequence[-1] * ratio
                    steps.append(f"Next number in sequence: {next_num}")
                    return next_num
                    
            # Polynomial fit
            steps.append("Attempting polynomial pattern recognition")
            for degree in range(1, min(4, len(sequence)-1)):
                try:
                    x = np.array(range(len(sequence)))
                    y = np.array(sequence)
                    coeffs = np.polyfit(x, y, degree)
                    next_x = len(sequence)
                    next_num = int(round(sum(c * (next_x ** i) for i, c in enumerate(coeffs))))
                    steps.append(f"Found polynomial pattern of degree {degree}")
                    steps.append(f"Next number in sequence: {next_num}")
                    return next_num
                except:
                    continue
        
        steps.append("Could not identify sequence pattern")
        return None
    
    def _solve_algebra(self, text: str, steps: List[str]) -> Any:
        """Solve algebraic problems"""
        try:
            # Extract equation parts
            equation_match = re.search(r'(\d+x\s*[+\-*/]\s*\d+\s*=\s*\d+)', text)
            if equation_match:
                equation = equation_match.group(1)
                steps.append(f"Found equation: {equation}")
                
                # Parse and solve equation
                x = symbols('x')
                parts = equation.split('=')
                lhs = simplify(parts[0])
                rhs = simplify(parts[1])
                solution = solve(Eq(lhs, rhs), x)
                
                if solution:
                    steps.append(f"Solution: x = {solution[0]}")
                    return float(solution[0])
            
            steps.append("No solvable equation found")
            return None
            
        except Exception as e:
            steps.append(f"Error solving algebra problem: {str(e)}")
            return None
    
    def _solve_geometry(self, text: str, steps: List[str]) -> Any:
        """Solve geometry problems"""
        try:
            if "area" in text.lower():
                # Try to find dimensions
                dimensions_match = re.search(r'(\d+)\s*[×x]\s*(\d+)', text)
                if dimensions_match:
                    length = float(dimensions_match.group(1))
                    width = float(dimensions_match.group(2))
                    area = length * width
                    steps.append(f"Calculated area: {length} × {width} = {area}")
                    return area
            
            elif "perimeter" in text.lower():
                # Similar handling for perimeter
                dimensions_match = re.search(r'(\d+)\s*[×x]\s*(\d+)', text)
                if dimensions_match:
                    length = float(dimensions_match.group(1))
                    width = float(dimensions_match.group(2))
                    perimeter = 2 * (length + width)
                    steps.append(f"Calculated perimeter: 2({length} + {width}) = {perimeter}")
                    return perimeter
            
            steps.append("Could not identify specific geometry calculation")
            return None
            
        except Exception as e:
            steps.append(f"Error solving geometry problem: {str(e)}")
            return None
    
    def _solve_general_math(self, text: str, numbers: List[str], expressions: List[str], steps: List[str]) -> Any:
        """Solve general mathematical problems"""
        try:
            if expressions:
                # Try to evaluate mathematical expressions
                expression = expressions[0].replace('x', '*')
                result = eval(expression)
                steps.append(f"Evaluated expression: {expression} = {result}")
                return result
            
            if numbers and len(numbers) >= 2:
                # Look for keywords to determine operation
                if any(word in text for word in ["sum", "total", "add"]):
                    result = sum(float(n) for n in numbers)
                    steps.append(f"Calculated sum: {result}")
                    return result
                elif any(word in text for word in ["difference", "subtract"]):
                    result = float(numbers[0]) - float(numbers[1])
                    steps.append(f"Calculated difference: {result}")
                    return result
                elif any(word in text for word in ["product", "multiply"]):
                    result = float(numbers[0]) * float(numbers[1])
                    steps.append(f"Calculated product: {result}")
                    return result
                elif any(word in text for word in ["divide", "quotient"]) and float(numbers[1]) != 0:
                    result = float(numbers[0]) / float(numbers[1])
                    steps.append(f"Calculated quotient: {result}")
                    return result
            
            steps.append("Could not determine appropriate mathematical operation")
            return None
            
        except Exception as e:
            steps.append(f"Error in general math solver: {str(e)}")
            return None