from typing import Dict, List, Any
import numpy as np
from .base_agent import BaseAgent

class Verifier(BaseAgent):
    """Agent responsible for verifying solutions and calculating confidence scores"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        super().__init__()
        self.confidence_threshold = confidence_threshold
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify solutions and calculate confidence scores
        
        Args:
            input_data (Dict[str, Any]): Contains the solutions to verify
            
        Returns:
            Dict[str, Any]: Verified solutions with confidence scores
        """
        solutions = input_data["solutions"]
        self._log_step("Verifying solutions", {"num_solutions": len(solutions)})
        
        verified_solutions = []
        for i, solution in enumerate(solutions):
            self._log_step(f"Verifying solution {i+1}")
            
            # Skip failed solutions
            if not solution["success"]:
                verified_solutions.append({
                    **solution,
                    "confidence": 0.0,
                    "verification_status": "failed"
                })
                continue
            
            try:
                confidence = self._verify_solution(solution)
                status = "verified" if confidence >= self.confidence_threshold else "uncertain"
                
                verified_solutions.append({
                    **solution,
                    "confidence": confidence,
                    "verification_status": status
                })
            except Exception as e:
                self.logger.error(f"Verification failed for solution {i+1}: {str(e)}")
                verified_solutions.append({
                    **solution,
                    "confidence": 0.0,
                    "verification_status": "error",
                    "error": str(e)
                })
        
        overall_confidence = self._calculate_overall_confidence(verified_solutions)
        
        return {
            "verified_solutions": verified_solutions,
            "overall_confidence": overall_confidence,
            "metadata": {
                "confidence_threshold": self.confidence_threshold,
                "verified_count": sum(1 for s in verified_solutions if s["verification_status"] == "verified")
            }
        }
    
    def _verify_solution(self, solution: Dict[str, Any]) -> float:
        """Calculate confidence score for a solution"""
        # Base confidence factors
        factors = []
        
        # Check solution presence
        if solution["solution"] is None:
            return 0.0
            
        # Tool-specific verification
        tool_type = solution["tool_used"]
        if tool_type == "math":
            factors.extend(self._verify_math_solution(solution))
        elif tool_type == "logic":
            factors.extend(self._verify_logic_solution(solution))
        else:
            factors.extend(self._verify_general_solution(solution))
        
        # Calculate final confidence
        if not factors:
            return 0.0
        return float(np.mean(factors))
    
    def _verify_math_solution(self, solution: Dict[str, Any]) -> List[float]:
        """Verify mathematical solutions"""
        factors = []
        
        # Check if solution is numeric
        if isinstance(solution["solution"], (int, float)):
            factors.append(1.0)
        
        # Check reasonable range
        try:
            value = float(solution["solution"])
            if -1e6 <= value <= 1e6:  # Reasonable range check
                factors.append(1.0)
            else:
                factors.append(0.5)
        except:
            factors.append(0.3)
        
        return factors
    
    def _verify_logic_solution(self, solution: Dict[str, Any]) -> List[float]:
        """Verify logical solutions"""
        factors = []
        
        # Check if solution is boolean for logical operations
        if isinstance(solution["solution"], bool):
            factors.append(1.0)
        
        # Check solution explanation
        if "reasoning" in solution and solution["reasoning"]:
            factors.append(0.8)
        
        return factors
    
    def _verify_general_solution(self, solution: Dict[str, Any]) -> List[float]:
        """Verify general solutions"""
        factors = []
        
        # Check solution is not empty
        if solution["solution"]:
            factors.append(0.7)
        
        # Check solution matches step format
        if isinstance(solution["solution"], (str, int, float, bool)):
            factors.append(0.8)
        
        return factors
    
    def _calculate_overall_confidence(self, solutions: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence across all solutions"""
        if not solutions:
            return 0.0
            
        confidences = [s["confidence"] for s in solutions]
        
        # Weight later steps slightly more as they often depend on earlier steps
        weights = np.linspace(0.8, 1.0, len(confidences))
        weighted_conf = np.average(confidences, weights=weights)
        
        return float(weighted_conf)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process solutions and verify them"""
        solution = input_data.get("solution", {})
        problem = input_data.get("problem", {})
        constraints = input_data.get("constraints", [])
        
        # Basic verification
        constraints_satisfied = self._check_constraints(solution, constraints)
        logic_valid = self._check_logic_validity(solution)
        completeness = self._check_completeness(solution, problem)
        
        overall_confidence = (constraints_satisfied + logic_valid + completeness) / 3.0
        
        return {
            "constraints_satisfied": constraints_satisfied,
            "logic_valid": logic_valid,
            "completeness": completeness,
            "overall_confidence": overall_confidence
        }
    
    def _check_constraints(self, solution: Dict[str, Any], constraints: List[str]) -> float:
        """Check if solution satisfies constraints"""
        if not constraints:
            return 1.0
        
        solution_text = str(solution.get("final_solution", "")).lower()
        satisfied = 0
        
        for constraint in constraints:
            # Simple keyword matching for constraint satisfaction
            constraint_words = constraint.lower().split()
            if any(word in solution_text for word in constraint_words):
                satisfied += 1
                
        return satisfied / len(constraints) if constraints else 1.0
    
    def _check_logic_validity(self, solution: Dict[str, Any]) -> float:
        """Check logical validity of solution"""
        # Basic check - solution exists and is not empty
        if solution.get("final_solution"):
            return 0.8
        return 0.0
    
    def _check_completeness(self, solution: Dict[str, Any], problem: Dict[str, Any]) -> float:
        """Check if solution is complete"""
        # Check if solution addresses the problem
        if solution.get("final_solution") and solution.get("explanation"):
            return 0.9
        elif solution.get("final_solution"):
            return 0.6
        return 0.0