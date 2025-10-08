import unittest
import time
from typing import Dict, Any, List
from main import AgentSystem
import json

class TestAgentSystem(unittest.TestCase):
    """Test suite for the Agent System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = AgentSystem()
        
        # Sample test problems
        self.test_problems = {
            "math": {
                "topic": "Mathematics",
                "problem_statement": "What is the next number in the sequence: 2, 4, 8, 16?",
                "expected_answer": "32"
            },
            "logic": {
                "topic": "Logic",
                "problem_statement": "If all A are B, and all B are C, what can we conclude about A and C?",
                "expected_answer": "All A are C"
            },
            "sequence": {
                "topic": "Sequence solving",
                "problem_statement": "What comes next: A, C, E, G?",
                "expected_answer": "I"
            }
        }
    
    def test_problem_decomposition(self):
        """Test if problems are correctly decomposed"""
        problem = self.test_problems["math"]
        analysis = self.agent._analyze_problem(problem)
        
        self.assertIn("subproblems", analysis)
        self.assertTrue(len(analysis["subproblems"]) > 0)
        
        # Check if numerical values are extracted
        self.assertIn("numerical_values", analysis)
        self.assertEqual(analysis["numerical_values"], [2, 4, 8, 16])
    
    def test_tool_selection(self):
        """Test if appropriate tools are selected"""
        for problem_type, problem in self.test_problems.items():
            tools = self.agent._select_tools(problem, self.agent._analyze_problem(problem))
            
            self.assertTrue(len(tools) > 0)
            
            if problem_type == "math":
                self.assertTrue(any(tool.name == "MathSolver" for tool in tools))
            elif problem_type == "logic":
                self.assertTrue(any(tool.name == "LogicSolver" for tool in tools))
            elif problem_type == "sequence":
                self.assertTrue(any(tool.name == "SequenceSolver" for tool in tools))
    
    def test_solution_verification(self):
        """Test solution verification process"""
        problem = self.test_problems["math"]
        solution = {
            "solution": "32",
            "reasoning_steps": [
                "Identified geometric sequence",
                "Common ratio is 2",
                "Next term = 16 × 2 = 32"
            ],
            "confidence": 0.9
        }
        
        verified = self.agent._verify_solution(problem, solution)
        
        self.assertIn("verification", verified)
        self.assertIn("confidence", verified)
        self.assertTrue(verified["confidence"] > 0)
    
    def test_error_handling(self):
        """Test error handling in the solution process"""
        # Create a malformed problem
        problem = {
            "topic": "Invalid",
            "problem_statement": ""  # Empty statement should cause error
        }
        
        result = self.agent.solve_problem(problem)
        
        self.assertIn("error", result)
        self.assertIn("partial_solution", result)
    
    def test_parallel_tool_execution(self):
        """Test parallel execution of tools"""
        problem = self.test_problems["math"]
        tools = self.agent._select_tools(problem, self.agent._analyze_problem(problem))
        
        results = self.agent._apply_tools(problem, tools)
        
        self.assertTrue(len(results) > 0)
        self.assertTrue(all("tool_name" in result for result in results))
    
    def test_output_formatting(self):
        """Test if output is correctly formatted"""
        # Process a batch of problems
        results = []
        for problem in self.test_problems.values():
            result = self.agent.solve_problem(problem)
            results.append(result)
        
        df = self.agent.format_output(results)
        
        self.assertEqual(len(df), len(self.test_problems))
        self.assertTrue(all(col in df.columns for col in ["id", "topic", "answer", "reasoning", "confidence"]))
    
    def test_caching(self):
        """Test if caching works correctly"""
        problem = self.test_problems["math"]
        
        # First call should not be cached
        start_time = time.time()
        first_result = self.agent._analyze_problem(problem)
        first_time = time.time() - start_time
        
        # Second call should be faster due to caching
        start_time = time.time()
        second_result = self.agent._analyze_problem(problem)
        second_time = time.time() - start_time
        
        self.assertLess(second_time, first_time)
        self.assertEqual(first_result, second_result)

if __name__ == '__main__':
    unittest.main()