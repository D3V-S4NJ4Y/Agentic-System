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