from typing import Dict, List, Any
import numpy as np
from sympy import symbols, solve, Eq
from ..symbolic_solver import SymbolicSolver

class SpatialSolver:
    """Handles spatial reasoning problems involving geometry, distances, and spatial relationships"""
    
    def __init__(self):
        self.symbolic = SymbolicSolver()
    
    def solve(self, problem: Dict[str, Any], steps: List[str]) -> Dict[str, Any]:
        text = problem["problem_statement"].lower()
        
        if "distance" in text or "equidistant" in text:
            return self._solve_distance_problem(text, steps)
        elif "angle" in text or "degree" in text:
            return self._solve_angle_problem(text, steps)
        elif "area" in text or "perimeter" in text:
            return self._solve_area_problem(text, steps)
        elif "volume" in text or "capacity" in text:
            return self._solve_volume_problem(text, steps)
        else:
            return self._solve_general_spatial(text, steps)
    
    def _solve_distance_problem(self, text: str, steps: List[str]) -> Dict[str, Any]:
        # Extract numbers and units
        distances = self.symbolic.extract_numbers(text)
        if not distances:
            return {"error": "No distances found in problem"}
            
        steps.append(f"Found distances: {distances}")
        
        # Handle equidistant points
        if "equidistant" in text:
            steps.append("Problem involves equidistant points")
            return {"result": self._find_equidistant_point(distances)}
            
        return {"result": self._calculate_distances(distances)}
    
    def _solve_angle_problem(self, text: str, steps: List[str]) -> Dict[str, Any]:
        angles = self.symbolic.extract_numbers(text)
        steps.append(f"Found angles: {angles}")
        
        if "complementary" in text:
            steps.append("Angles are complementary (sum to 90°)")
            return {"result": 90 - sum(angles)}
        elif "supplementary" in text:
            steps.append("Angles are supplementary (sum to 180°)")
            return {"result": 180 - sum(angles)}
            
        return {"result": self._calculate_angles(angles)}
    
    def _solve_area_problem(self, text: str, steps: List[str]) -> Dict[str, Any]:
        dimensions = self.symbolic.extract_numbers(text)
        steps.append(f"Found dimensions: {dimensions}")
        
        if len(dimensions) < 2:
            return {"error": "Insufficient dimensions for area calculation"}
            
        if "triangle" in text:
            area = 0.5 * dimensions[0] * dimensions[1]
            steps.append(f"Calculating triangle area: 1/2 * base * height")
            return {"result": area}
            
        area = dimensions[0] * dimensions[1]
        steps.append(f"Calculating rectangular area: length * width")
        return {"result": area}
    
    def _solve_volume_problem(self, text: str, steps: List[str]) -> Dict[str, Any]:
        dimensions = self.symbolic.extract_numbers(text)
        steps.append(f"Found dimensions: {dimensions}")
        
        if len(dimensions) < 3:
            return {"error": "Insufficient dimensions for volume calculation"}
            
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        steps.append(f"Calculating volume: length * width * height")
        return {"result": volume}
    
    def _solve_general_spatial(self, text: str, steps: List[str]) -> Dict[str, Any]:
        # Use symbolic solver for complex spatial relationships
        equations = self.symbolic.extract_equations(text)
        if equations:
            steps.append(f"Found equations: {equations}")
            solution = self.symbolic.solve_equations(equations)
            return {"result": solution}
            
        return {"error": "Could not determine spatial relationship"}
    
    def _find_equidistant_point(self, points: List[float]) -> float:
        """Find point equidistant from given points"""
        return sum(points) / len(points)
    
    def _calculate_distances(self, distances: List[float]) -> float:
        """Calculate total distance or average"""
        if len(distances) == 1:
            return distances[0]
        return sum(distances)
    
    def _calculate_angles(self, angles: List[float]) -> float:
        """Calculate missing angle or total"""
        if len(angles) == 1:
            return angles[0]
        return sum(angles)