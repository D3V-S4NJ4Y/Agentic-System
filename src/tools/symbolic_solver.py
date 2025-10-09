from typing import Dict, Any, List
import sympy as sp
import re

class SymbolicSolver:
    """Symbolic mathematics solver using SymPy"""
    
    def __init__(self):
        self.symbols = {}
        
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve mathematical problems symbolically"""
        try:
            problem_text = problem.get("problem_statement", "")
            
            # Extract equations and expressions
            equations = self._extract_equations(problem_text)
            variables = self._extract_variables(problem_text)
            
            if equations:
                # Solve system of equations
                solutions = sp.solve(equations, variables)
                return {
                    "solution": solutions,
                    "method": "symbolic_equation_solving",
                    "equations": str(equations),
                    "variables": str(variables)
                }
            else:
                # Try to evaluate expressions
                expressions = self._extract_expressions(problem_text)
                if expressions:
                    results = [sp.simplify(expr) for expr in expressions]
                    return {
                        "solution": results,
                        "method": "symbolic_evaluation",
                        "expressions": str(expressions)
                    }
                    
            return {"solution": None, "error": "No mathematical expressions found"}
            
        except Exception as e:
            return {"solution": None, "error": str(e)}
    
    def _extract_equations(self, text: str) -> List[sp.Eq]:
        """Extract equations from text"""
        equations = []
        # Look for patterns like "x + 2 = 5"
        eq_patterns = re.findall(r'([^=]+)=([^=]+)', text)
        
        for left, right in eq_patterns:
            try:
                left_expr = sp.sympify(left.strip())
                right_expr = sp.sympify(right.strip())
                equations.append(sp.Eq(left_expr, right_expr))
            except:
                continue
                
        return equations
    
    def _extract_variables(self, text: str) -> List[sp.Symbol]:
        """Extract variables from text"""
        # Common variable names
        var_names = re.findall(r'\b[a-zA-Z]\b', text)
        return [sp.Symbol(name) for name in set(var_names)]
    
    def _extract_expressions(self, text: str) -> List[sp.Basic]:
        """Extract mathematical expressions from text"""
        expressions = []
        # Look for mathematical expressions
        expr_patterns = re.findall(r'[\d\+\-\*/\^\(\)x-z]+', text)
        
        for expr in expr_patterns:
            try:
                expressions.append(sp.sympify(expr))
            except:
                continue
                
        return expressions