from typing import Dict, List, Any, Optional
import re
from .base_agent import BaseAgent
from ..tools.symbolic_solver import SymbolicSolver
from ..tools.calculator import Calculator
from ..tools.pattern_matcher import PatternMatcher

class Solver(BaseAgent):
    """Agent responsible for solving decomposed problem steps"""
    
    def __init__(self):
        super().__init__()
        self.tools = {
            "math": SymbolicSolver(),
            "logic": PatternMatcher(),
            "calculator": Calculator()
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process each step and generate solutions with high accuracy
        
        Args:
            input_data (Dict[str, Any]): Contains the decomposed steps
            
        Returns:
            Dict[str, Any]: Solutions for each step with verification
        """
        steps = input_data.get("steps", [])
        problem_type = input_data.get("type", "unknown")
        self._log_step("Processing steps", {"num_steps": len(steps), "type": problem_type})
        
        solutions = []
        context = {}  # Maintain context between steps
        
        for i, step in enumerate(steps):
            self._log_step(f"Solving step {i+1}", {"step": step})
            
            # Select most appropriate tool based on step type and previous context
            tool_type = self._select_tool(step, context)
            tool = self.tools[tool_type]
            
            try:
                # Pre-process step with context
                enriched_step = self._enrich_with_context(step, context)
                
                # Solve step
                solution = tool.solve(enriched_step)
                
                # Verify solution
                verification = self._verify_solution(solution, step)
                
                # Update context with new information
                context.update(self._extract_context(solution))
                
                solutions.append({
                    "step": step,
                    "tool_used": tool_type,
                    "solution": solution,
                    "verification": verification,
                    "confidence": self._calculate_confidence(verification),
                    "success": True
                })
            except Exception as e:
                self.logger.error(f"Failed to solve step {i+1}: {str(e)}")
                solutions.append({
                    "step": step,
                    "tool_used": tool_type,
                    "solution": None,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "steps": steps,
            "solutions": solutions,
            "metadata": {
                "success_rate": self._calculate_success_rate(solutions)
            }
        }
    
    def _select_tool(self, step: str, context: Dict[str, Any]) -> str:
        """Select the appropriate tool for solving a step"""
        step = step.lower()
        
        # Check for mathematical expressions
        if re.search(r'[\d+\-*/=]', step) or any(word in step for word in ["calculate", "solve", "evaluate"]):
            return "math"
            
        # Check for logical reasoning
        if any(word in step for word in ["if", "then", "or", "and", "not", "implies"]):
            return "logic"
            
        # Default to calculator for numerical operations
        if re.search(r'\d+', step):
            return "calculator"
            
        return "logic"  # Default to logical reasoning
    
    def _enrich_with_context(self, step: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich step with context from previous solutions"""
        return {
            "problem_statement": step,
            "context": context
        }
    
    def _verify_solution(self, solution: Dict[str, Any], step: str) -> Dict[str, Any]:
        """Verify a solution for a step"""
        if solution.get("solution") is not None:
            return {"valid": True, "confidence": 0.8}
        else:
            return {"valid": False, "confidence": 0.0}
    
    def _extract_context(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context information from solution"""
        context = {}
        if "solution" in solution and solution["solution"] is not None:
            context["last_result"] = solution["solution"]
        return context
    
    def _calculate_confidence(self, verification: Dict[str, Any]) -> float:
        """Calculate confidence score from verification"""
        return verification.get("confidence", 0.0)
    
    def _calculate_success_rate(self, solutions: List[Dict[str, Any]]) -> float:
        """Calculate the success rate of solutions"""
        if not solutions:
            return 0.0
        successful = sum(1 for s in solutions if s["success"])
        return successful / len(solutions)