# Agentic Reasoning System

An intelligent system designed to solve complex logic and reasoning problems through autonomous decomposition, tool selection, and solution verification. This system is optimized for performance and accuracy, with extensive testing and verification capabilities.

## Overview

The Agentic Reasoning System is designed to tackle complex reasoning problems by breaking them down into manageable subproblems, selecting appropriate tools for each subproblem, executing the solutions in parallel when possible, and providing transparent reasoning traces. The system features advanced caching, comprehensive verification, and detailed documentation of each reasoning step.

## System Architecture

The system follows a modular architecture with the following components:

1. **Problem Analyzer**: Analyzes the problem to identify its type, key elements, and constraints.
2. **Tool Selector**: Selects appropriate tools based on the problem analysis.
3. **Tool Executor**: Applies the selected tools to solve the problem.
4. **Result Integrator**: Integrates the results from different tools.
5. **Solution Verifier**: Verifies the solution for correctness and consistency.
6. **Answer Selector**: Selects the most likely answer option based on the solution.

## Specialized Solvers

The system includes several specialized solvers for different types of problems:

1. **MathSolver**: Solves mathematical problems using symbolic math.
2. **LogicSolver**: Solves logical reasoning problems.
3. **SpatialReasoningSolver**: Solves spatial reasoning problems.
4. **OptimizationSolver**: Solves optimization problems.
5. **LateralThinkingSolver**: Solves lateral thinking problems.
6. **MechanismSolver**: Solves problems about mechanisms and their operations.
7. **ClassicRiddleSolver**: Solves classic riddles.
8. **SequenceSolver**: Solves sequence problems.

## Problem Decomposition

The system decomposes complex problems into simpler subproblems by:

1. Analyzing the problem statement to identify the problem type.
2. Breaking down the problem into key components.
3. Identifying the constraints and requirements.
4. Determining the appropriate solution approach.

## Tool Selection

The system selects appropriate tools based on:

1. The problem type (e.g., mathematical, logical, spatial).
2. The presence of specific keywords in the problem statement.
3. The constraints and requirements of the problem.

## Execution & Verification

The system executes the selected tools and verifies the results by:

1. Applying the tools to solve the subproblems.
2. Integrating the results from different tools.
3. Checking if the solution addresses the question.
4. Verifying the internal consistency of the solution.
5. Assessing the plausibility of the solution.

## Reasoning Traces

The system provides transparent reasoning traces that include:

1. The problem analysis.
2. The selected tools.
3. The step-by-step reasoning process.
4. The verification steps.
5. The final solution.

## Installation and Usage

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. run the system:
```bash
python main.py
```

5. Use the system:
```python
from main import AgentSystem

# Initialize the agent system
agent = AgentSystem()

# Solve a single problem
problem = {
    "topic": "Mathematics",
    "problem_statement": "What is the next number in the sequence: 2, 4, 8, 16?"
}
solution = agent.solve_problem(problem)
print(solution)

# Process a dataset
agent.process_dataset("test.csv", "output.csv")
```

## Project Structure

```
├── main.py              # Main agent system implementation
├── utils/
│   ├── __init__.py
│   ├── agent_utils.py   # Agent helper functions
│   ├── cache_utils.py   # Caching utilities
│   ├── decomposition_utils.py  # Problem decomposition
│   └── verification_utils.py   # Solution verification
├── tests/
│   └── test_agent_system.py   # Test suite
├── spatial_reasoning.py   # Specialized spatial reasoning solver
├── sequence_solver.py     # Specialized sequence solver
├── optimization_solver.py # Specialized optimization solver
├── lateral_thinking.py    # Specialized lateral thinking solver
├── classic_riddles.py     # Specialized riddle solver
├── mechanism_solver.py    # Specialized mechanism solver
├── requirements.txt      # Project dependencies
└── README.md            # Documentation

## Performance

The system achieves strong results on the given dataset by:

1. Accurately identifying the problem type.
2. Selecting appropriate tools for each problem.
3. Providing detailed reasoning traces.
4. Verifying the solutions for correctness and consistency.

## Limitations and Future Work

The current system has some limitations:

1. It relies on pattern matching and rule-based approaches for many problem types.
2. It may struggle with problems that require deep domain knowledge.
3. The verification process could be more sophisticated.

Future work could include:

1. Incorporating more advanced symbolic reasoning capabilities.
2. Enhancing the problem decomposition process.
3. Improving the verification process.
4. Adding more specialized solvers for different problem types.

## Conclusion

The Agentic Reasoning System demonstrates the power of combining modular problem-solving approaches with transparent reasoning traces. By breaking down complex problems into manageable subproblems and applying appropriate tools, the system can solve a wide range of reasoning problems effectively...
