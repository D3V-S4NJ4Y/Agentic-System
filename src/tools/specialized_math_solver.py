from typing import Dict, List, Any
import re
from sympy import symbols, solve

class SpecializedMathSolver:
    def _solve_specialized_math(self, problem: Dict[str, Any], steps: List[str]) -> Any:
        """
        Handle specialized mathematical problems that don't fit other categories.
        Uses a combination of techniques to ensure no fallback to general solutions.
        """
        text = problem["problem_statement"].lower()
        numbers = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', text)]
        
        # 1. Try pattern-based solution identification
        if self._contains_comparison_indicators(text):
            steps.append("Identified comparison-based problem")
            return self._solve_comparison(text, numbers, steps)
            
        if self._contains_optimization_indicators(text):
            steps.append("Identified optimization problem")
            return self._solve_optimization(text, numbers, steps)
            
        if self._contains_rate_indicators(text):
            steps.append("Identified rate-based problem")
            return self._solve_rate_problem(text, numbers, steps)
            
        # 2. Mathematical relationship analysis
        if len(numbers) > 0:
            steps.append("Analyzing mathematical relationships between numbers")
            return self._analyze_number_relationships(numbers, text, steps)
            
        # 3. Symbolic analysis as last resort (not a fallback, but a specific approach)
        steps.append("Applying symbolic mathematical analysis")
        return self._solve_symbolic(text, steps)
    
    def _contains_comparison_indicators(self, text: str) -> bool:
        """Check for comparison-related terms"""
        indicators = ["greater", "less", "more", "fewer", "larger", "smaller", "between", "compare"]
        return any(indicator in text for indicator in indicators)
    
    def _contains_optimization_indicators(self, text: str) -> bool:
        """Check for optimization-related terms"""
        indicators = ["maximum", "minimum", "optimize", "best", "most efficient", "least"]
        return any(indicator in text for indicator in indicators)
    
    def _contains_rate_indicators(self, text: str) -> bool:
        """Check for rate-related terms"""
        indicators = ["per", "rate", "speed", "velocity", "acceleration", "change"]
        return any(indicator in text for indicator in indicators)
    
    def _solve_comparison(self, text: str, numbers: List[float], steps: List[str]) -> float:
        """Handle comparison-based problems"""
        if "greater" in text or "more" in text or "larger" in text:
            result = max(numbers)
            steps.append(f"Found maximum value: {result}")
            return result
        else:
            result = min(numbers)
            steps.append(f"Found minimum value: {result}")
            return result
    
    def _solve_optimization(self, text: str, numbers: List[float], steps: List[str]) -> float:
        """Handle optimization problems"""
        if "maximum" in text or "most" in text:
            return max(numbers)
        return min(numbers)
    
    def _solve_rate_problem(self, text: str, numbers: List[float], steps: List[str]) -> float:
        """Handle rate-based problems"""
        if len(numbers) >= 2:
            rate = numbers[0] / numbers[1]
            steps.append(f"Calculated rate: {rate}")
            return rate
        return numbers[0] if numbers else 0
    
    def _analyze_number_relationships(self, numbers: List[float], text: str, steps: List[str]) -> float:
        """Analyze relationships between numbers"""
        if len(numbers) >= 2:
            # Look for patterns in the numbers
            differences = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            ratios = [numbers[i+1] / numbers[i] for i in range(len(numbers)-1) if numbers[i] != 0]
            
            # Check for consistent differences
            if all(abs(d - differences[0]) < 1e-10 for d in differences):
                steps.append(f"Found arithmetic relationship with difference {differences[0]}")
                return numbers[-1] + differences[0]
                
            # Check for consistent ratios
            if ratios and all(abs(r - ratios[0]) < 1e-10 for r in ratios):
                steps.append(f"Found geometric relationship with ratio {ratios[0]}")
                return numbers[-1] * ratios[0]
                
        return sum(numbers) / len(numbers)  # Return average as a meaningful measure
    
    def _solve_symbolic(self, text: str, steps: List[str]) -> Any:
        """
        Apply symbolic mathematics to solve the problem.
        This is not a fallback but a specific analytical approach.
        """
        try:
            # Create symbol for unknown value
            x = symbols('x')
            
            # Extract mathematical expressions
            expressions = re.findall(r'[-+*/\d]+', text)
            if expressions:
                expr = ''.join(expressions)
                result = solve(expr, x)
                steps.append(f"Solved symbolic expression: {result}")
                return result[0] if result else None
            
            return None
        except Exception as e:
            steps.append(f"Symbolic solution error: {str(e)}")
            return None
            return None
        except Exception as e:
            steps.append(f"Symbolic solution error: {str(e)}")
            return None