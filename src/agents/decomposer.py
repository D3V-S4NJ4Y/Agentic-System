from typing import Dict, List, Any, Tuple
import re
from .base_agent import BaseAgent

class Decomposer(BaseAgent):
    """Enhanced decomposer agent with improved problem breakdown capabilities"""
    
    def __init__(self):
        super().__init__()
        self.decomposition_patterns = {
            'sequence': [
                ('identify', r'(?:find|what|determine)\s+(?:next|following|pattern)'),
                ('extract', r'(\d+(?:\s*,\s*\d+)+)'),
                ('requirements', r'(?:must|should|needs to|have to)'),
                ('constraints', r'(?:only|at most|at least|between)')
            ],
            'spatial': [
                ('position', r'(?:where|position|location|place)'),
                ('movement', r'(?:moves|goes|travels|turns)'),
                ('arrangement', r'(?:arrange|order|organize)'),
                ('dimensions', r'(\d+\s*(?:x|×)\s*\d+(?:\s*(?:x|×)\s*\d+)?)')
            ],
            'mechanism': [
                ('components', r'(?:gear|pulley|lever|wheel)s?'),
                ('action', r'(?:rotate|turn|pull|push|lift)s?'),
                ('measurement', r'(\d+\s*(?:kg|m|cm|N|newton))'),
                ('efficiency', r'(\d+\s*%)')
            ]
        }
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced problem decomposition with pattern recognition and subtask generation
        
        Args:
            input_data (Dict[str, Any]): Contains the problem statement and context
            
        Returns:
            Dict[str, Any]: Decomposed steps, patterns, and analysis results
        """
        problem = input_data["problem"]
        self._log_step("Starting problem decomposition")
        
        # Identify problem types
        problem_types = self.classify_problem(problem)
        self._log_step("Problem types identified", {"types": problem_types})
        
        # Extract key components
        components = self._extract_components(problem, problem_types)
        self._log_step("Components extracted", components)
        
        # Generate decomposition steps
        steps = []
        subtasks = []
        
        # For each identified type, add relevant decomposition steps
        for p_type in problem_types:
            type_steps, type_subtasks = self._decompose_by_type(problem, p_type, components)
            steps.extend(type_steps)
            subtasks.extend(type_subtasks)
            
        # Add verification steps
        verification_steps = self._generate_verification_steps(problem, problem_types)
        steps.extend(verification_steps)
        
        result = {
            "original_problem": problem,
            "problem_types": problem_types,
            "components": components,
            "steps": steps,
            "subtasks": subtasks,
            "metadata": {
                "decomposition_method": "pattern_based",
                "num_steps": len(steps),
                "num_subtasks": len(subtasks),
                "verification_steps": len(verification_steps)
            }
        }
        
        self._log_step("Decomposition complete", {
            "num_steps": len(steps),
            "num_subtasks": len(subtasks)
        })
        return result
    
    def _extract_components(self, problem: str, problem_types: List[str]) -> Dict[str, Any]:
        """Extract key components based on problem types"""
        components = {}
        
        for p_type in problem_types:
            if p_type in self.decomposition_patterns:
                type_components = {}
                patterns = self.decomposition_patterns[p_type]
                
                for name, pattern in patterns:
                    matches = re.findall(pattern, problem, re.IGNORECASE)
                    if matches:
                        type_components[name] = matches
                        
                if type_components:
                    components[p_type] = type_components
                    
        return components
    
    def _decompose_by_type(self, problem: str, problem_type: str, components: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Enhanced problem decomposition with subtask generation"""
        steps = []
        subtasks = []
        
        type_components = components.get(problem_type, {})
        
        if problem_type == "sequence":
            if 'extract' in type_components:
                sequence_str = type_components['extract'][0]
                steps.extend([
                    f"1. Analyze sequence: {sequence_str}",
                    "2. Determine sequence characteristics:",
                    "   - Pattern identification",
                    "   - Arithmetic/geometric progression check",
                    "   - Special sequence detection",
                    "   - Look for polynomial patterns",
                    "   - Examine special sequences"
                ])
                subtasks.extend([
                    {
                        "type": "sequence_analysis", 
                        "data": {
                            "sequence": sequence_str,
                            "patterns": type_components.get('identify', []),
                            "constraints": type_components.get('constraints', [])
                        }
                    },
                    {
                        "type": "pattern_detection",
                        "data": {
                            "sequence": sequence_str,
                            "requirements": type_components.get('requirements', [])
                        }
                    },
                    {
                        "type": "next_term_prediction",
                        "data": {
                            "sequence": sequence_str,
                            "identified_patterns": type_components.get('identify', [])
                        }
                    }
                ])
                
        elif problem_type == "spatial":
            position_info = type_components.get('position', [])
            movement_info = type_components.get('movement', [])
            steps.extend([
                "1. Analyze spatial configuration:",
                f"   - Initial positions: {', '.join(position_info) if position_info else 'Not specified'}",
                "2. Track transformations:",
                f"   - Movement patterns: {', '.join(movement_info) if movement_info else 'Not specified'}",
                "   - Record direction changes",
                "   - Calculate distances",
                "   - Apply spatial transformations"
            ])
            subtasks.extend([
                {
                    "type": "position_tracking",
                    "data": {
                        "positions": position_info,
                        "arrangement": type_components.get('arrangement', [])
                    }
                },
                {
                    "type": "movement_analysis",
                    "data": {
                        "movements": movement_info,
                        "dimensions": type_components.get('dimensions', [])
                    }
                }
            ])
                
        elif problem_type == "mechanism":
            components_info = type_components.get('components', [])
            action_info = type_components.get('action', [])
            steps.extend([
                "1. Analyze mechanical system:",
                f"   - Components: {', '.join(components_info) if components_info else 'Not specified'}",
                f"   - Actions: {', '.join(action_info) if action_info else 'Not specified'}",
                "2. Process system dynamics:",
                "   - Calculate mechanical advantage",
                "   - Determine system efficiency",
                "   - Analyze energy transfer"
            ])
            subtasks.extend([
                {
                    "type": "mechanical_analysis",
                    "data": {
                        "components": components_info,
                        "actions": action_info
                    }
                },
                {
                    "type": "efficiency_calculation",
                    "data": {
                        "system_components": components_info
                    }
                }
            ])
                
        return steps, subtasks
    
    def _generate_verification_steps(self, problem: str, problem_types: List[str]) -> List[str]:
        """
        Generate verification steps based on problem types
        
        Args:
            problem (str): The original problem text
            problem_types (List[str]): List of identified problem types
            
        Returns:
            List[str]: List of verification steps to perform
        """
        verification_steps = [
            "Final Verification Steps:",
            "1. General validation:"
        ]
        
        for p_type in problem_types:
            if p_type == "sequence":
                verification_steps.extend([
                    "2. Sequence verification:",
                    "   - Verify pattern consistency",
                    "   - Check term predictions",
                    "   - Validate pattern rules"
                ])
            elif p_type == "spatial":
                verification_steps.extend([
                    "2. Spatial verification:",
                    "   - Validate position tracking",
                    "   - Confirm transformations",
                    "   - Check final configuration"
                ])
            elif p_type == "mechanism":
                verification_steps.extend([
                    "2. Mechanical verification:",
                    "   - Check system dynamics",
                    "   - Verify energy conservation",
                    "   - Validate efficiency"
                ])
        
        # Add general verification steps
        verification_steps.extend([
            "3. Cross-validation:",
            "   - Check solution completeness",
            "   - Verify requirement satisfaction",
            "   - Validate edge cases"
        ])
        
        return verification_steps