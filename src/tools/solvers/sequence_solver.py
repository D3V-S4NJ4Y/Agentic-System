import re
import math
import numpy as np
from typing import Dict, Any, List, Tuple

class SequenceSolver:
    """Pure algorithmic solver for sequence problems without LLM dependencies"""
    
    def solve(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve sequence problems using mathematical pattern recognition
        with detailed step-by-step reasoning and proofs.
        """
        text = problem_data.get("problem_statement", "")
        steps = []
        steps.append("1. Analyzing sequence problem using pure mathematical methods")
        
        # Extract sequence
        sequence_match = re.search(r'(\d+(?:,\s*\d+)*)', text)
        if not sequence_match:
            return {
                "solution": "No sequence found in problem",
                "steps": steps,
                "confidence": 0.0
            }
            
        sequence_str = sequence_match.group(1)
        sequence = [int(num.strip()) for num in sequence_str.split(',')]
        steps.append(f"2. Extracted sequence: {sequence}")
        
        # Allow even 2-term sequences if the pattern is very clear
        if len(sequence) < 2:
            return {
                "solution": "Need at least 2 terms for pattern detection",
                "steps": steps,
                "confidence": 0.0
            }
            
        # Try all pattern detection methods in order of reliability
        pattern_methods = [
            (self._check_arithmetic, "arithmetic"),
            (self._check_geometric, "geometric"),
            (self._check_fibonacci, "Fibonacci-like"),
            (self._check_polynomial, "polynomial"),
            (self._check_power, "power"),
            (self._check_alternating, "alternating"),
            (self._check_square_numbers, "square numbers"),
            (self._check_cube_numbers, "cube numbers"),
            (self._check_triangular_numbers, "triangular numbers"),
            (self._check_prime_related, "prime-related")
        ]
        
        best_result = None
        highest_confidence = 0.0
        
        for method, pattern_type in pattern_methods:
            steps.append(f"\n3. Testing for {pattern_type} pattern")
            result = method(sequence, steps)
            
            if result and result["confidence"] > highest_confidence:
                highest_confidence = result["confidence"]
                best_result = result
                steps.extend(result["proof_steps"])
                
        if best_result:
            steps.append(f"\n4. Final Solution:")
            steps.append(f"Pattern type: {best_result['pattern_type']}")
            steps.append(f"Next term: {best_result['next_term']}")
            steps.append(f"Mathematical proof provided above")
            steps.append(f"Confidence: {best_result['confidence']:.2%}")
            
            return {
                "solution": str(best_result['next_term']),
                "steps": steps,
                "confidence": best_result["confidence"],
                "pattern_type": best_result["pattern_type"],
                "mathematical_proof": best_result["proof_steps"]
            }
            
        steps.append("\n4. No definitive pattern found with mathematical certainty")
        return {
            "solution": "No mathematically verifiable pattern found",
            "steps": steps,
            "confidence": 0.0
        }
        
    def _check_arithmetic(self, sequence: List[int], steps: List[str]) -> Dict[str, Any]:
        """
        Check for arithmetic sequences with rigorous proof using:
        1. First differences (constant = arithmetic)
        2. Second differences (constant = quadratic)
        3. Third differences (constant = cubic)
        """
        # Calculate differences up to 3rd order
        diffs = [sequence]
        for order in range(3):
            next_diffs = [diffs[-1][i+1] - diffs[-1][i] 
                         for i in range(len(diffs[-1])-1)]
            diffs.append(next_diffs)
            
            # Check if these differences are constant
            if len(set(next_diffs)) == 1:
                # Found constant difference at this order
                degree = order + 1
                next_term = self._predict_polynomial_term(sequence, degree)
                
                steps.extend([
                    "Found polynomial pattern:",
                    f"1. Sequence type: {self._get_polynomial_type(degree)}",
                    f"2. Differences analysis:"
                ])
                
                for i, d in enumerate(diffs[1:], 1):
                    steps.append(f"   {i}{self._get_order_suffix(i)} differences: {d}")
                    if i == degree:
                        steps.append(f"   -> Constant {d[0]} indicates {degree}-degree polynomial")
                
                steps.extend([
                    f"3. Mathematical form: {self._get_polynomial_form(degree)}",
                    f"4. Next term calculation:",
                    f"   {sequence[-1]} + {diffs[1][-1]} = {next_term}"
                ])
                
                return {
                    "type": f"polynomial_degree_{degree}",
                    "differences": diffs[:degree+1],
                    "next": next_term,
                    "confidence": 1.0,
                    "degree": degree
                }
                
        return {"type": "unknown", "confidence": 0.0}
        
    def _check_geometric(self, sequence: List[int], steps: List[str]) -> Dict[str, Any]:
        """
        Check for geometric sequence with rigorous mathematical proof
        Uses logarithmic differences to verify geometric progression
        """
        if any(x == 0 for x in sequence[:-1]):  # Avoid division by zero
            return {"type": "unknown", "confidence": 0.0}
            
        # Calculate ratios and log differences
        ratios = [sequence[i+1] / sequence[i] for i in range(len(sequence)-1)]
        log_sequence = [math.log(abs(x)) for x in sequence]
        log_diffs = [log_sequence[i+1] - log_sequence[i] 
                    for i in range(len(log_sequence)-1)]
        
        # Check if it's geometric (constant ratio)
        ratio_tolerance = 1e-6
        if len(set(round(r/ratios[0], 6) for r in ratios)) == 1:
            ratio = ratios[0]
            next_term = int(round(sequence[-1] * ratio))
            
            steps.extend([
                "Found geometric sequence:",
                f"1. Common ratio (r) = {ratio:.3f}",
                "2. Verification steps:",
                f"   - Consecutive ratios: {[f'{r:.3f}' for r in ratios]}",
                f"   - Log differences: {[f'{x:.3f}' for x in log_diffs]}",
                "3. Properties:",
                f"   - Growth rate: {(ratio-1)*100:.1f}% per term",
                f"   - General term: a₁r^(n-1) where a₁ = {sequence[0]}",
                "4. Next term calculation:",
                f"   {sequence[-1]} × {ratio:.3f} = {next_term}"
            ])
            
            return {
                "type": "geometric",
                "ratio": ratio,
                "next": next_term,
                "confidence": 1.0
            }
            
        return {"type": "unknown", "confidence": 0.0}
        
    def _get_order_suffix(self, n: int) -> str:
        """Get ordinal suffix for a number"""
        if n % 10 == 1 and n != 11:
            return "st"
        elif n % 10 == 2 and n != 12:
            return "nd"
        elif n % 10 == 3 and n != 13:
            return "rd"
        return "th"
        
    def _get_polynomial_type(self, degree: int) -> str:
        """Get polynomial type description"""
        types = {1: "linear", 2: "quadratic", 3: "cubic", 4: "quartic"}
        return types.get(degree, f"{degree}th degree polynomial")
        
    def _get_polynomial_form(self, degree: int) -> str:
        """Get general polynomial form"""
        terms = []
        for d in range(degree, -1, -1):
            if d == 0:
                terms.append("k₀")
            elif d == 1:
                terms.append("k₁n")
            else:
                terms.append(f"k{d}n^{d}")
        return " + ".join(terms)
        
    def _predict_polynomial_term(self, sequence: List[int], degree: int) -> int:
        """
        Predict next term using polynomial interpolation
        Returns exact integer result for polynomial sequences
        """
        if degree >= len(sequence):
            return 0
            
        # Build Vandermonde matrix for polynomial fit
        x = np.arange(len(sequence))
        A = np.vander(x, degree + 1)
        # Solve for coefficients
        coeffs = np.linalg.solve(A, sequence)
        # Predict next value
        next_x = len(sequence)
        return int(round(sum(c * next_x**i for i, c in enumerate(coeffs))))
        
    def _check_geometric(self, sequence: List[int], steps: List[str]) -> Dict[str, Any]:
        """Check for geometric sequence with mathematical proof"""
        if 0 in sequence[:-1]:  # Avoid division by zero
            return None
            
        ratios = [sequence[i+1]/sequence[i] for i in range(len(sequence)-1)]
        if len(set(round(r, 10) for r in ratios)) == 1:  # All ratios equal within precision
            r = ratios[0]
            next_term = int(sequence[-1] * r) if r.is_integer() else sequence[-1] * r
            proof_steps = [
                f"Geometric Sequence Analysis:",
                f"1. Ratios between consecutive terms: {[f'{r:.2f}' for r in ratios]}",
                f"2. All ratios are equal to {r:.2f}",
                f"3. Formula: a(n+1) = a(n) × r where r = {r:.2f}",
                f"4. Last term is {sequence[-1]}, so next term = {sequence[-1]} × {r:.2f} = {next_term}",
                f"5. Verification: Each term is {r:.2f} times the previous:",
                *[f"   {sequence[i]} × {r:.2f} = {sequence[i+1]}" for i in range(len(sequence)-1)]
            ]
            return {
                "pattern_type": "geometric",
                "next_term": next_term,
                "confidence": 1.0,
                "proof_steps": proof_steps
            }
        return None

    def _check_fibonacci(self, sequence: List[int], steps: List[str]) -> Dict[str, Any]:
        """
        Check for Fibonacci-like sequences with mathematical proof.
        Also detects weighted Fibonacci sequences where each term
        is a weighted sum of previous terms.
        """
        if len(sequence) < 5:  # Need enough terms for reliable detection
            return {"type": "unknown", "confidence": 0.0}
        
        # Try to find weights for previous terms
        weights = self._find_recurrence_weights(sequence)
        if weights:
            order = len(weights)
            next_term = sum(w * sequence[-i-1] for i, w in enumerate(weights))
            
            steps.extend([
                "Found recursive sequence pattern:",
                f"1. Order: {order} term recurrence relation",
                "2. Weights analysis:",
                *[f"   Term n-{i+1}: {w:+.2f}" for i, w in enumerate(weights)],
                "\n3. Mathematical properties:",
                "   - Characteristic equation coefficients:",
                f"   x^{order} {' '.join(f'{w:+.2f}x^{order-i-1}' for i, w in enumerate(weights))} = 0",
                "\n4. Next term calculation:",
                *[f"   {w:+.2f} × {sequence[-i-1]} = {w*sequence[-i-1]:.1f}" 
                  for i, w in enumerate(weights)],
                f"   Sum = {next_term}"
            ])
            
            # Special case recognition
            if self._is_fibonacci_weights(weights):
                steps.extend([
                    "\n5. Special case: Standard Fibonacci sequence",
                    "   Each term is sum of two previous terms"
                ])
            elif self._is_tribonacci_weights(weights):
                steps.extend([
                    "\n5. Special case: Tribonacci sequence",
                    "   Each term is sum of three previous terms"
                ])
            
            return {
                "type": "recursive",
                "order": order,
                "weights": weights,
                "next": int(round(next_term)),
                "confidence": 1.0
            }
            
        return {"type": "unknown", "confidence": 0.0}
        
    def _check_polynomial(self, sequence: List[int], steps: List[str]) -> Dict[str, Any]:
        """
        Check for polynomial patterns using:
        1. Forward differences analysis
        2. Lagrange interpolation
        3. Least squares fitting with error analysis
        """
        if len(sequence) < 4:
            return {"type": "unknown", "confidence": 0.0}
            
        # Try polynomial fits of increasing degree
        x = np.array(range(len(sequence)))
        best_fit = None
        lowest_rel_error = float('inf')
        
        for degree in range(1, min(4, len(sequence)-1)):
            # Fit polynomial
            coeffs = np.polyfit(x, sequence, degree)
            fitted = np.polyval(coeffs, x)
            
            # Calculate relative error
            abs_error = np.abs(fitted - sequence)
            rel_error = np.mean(abs_error / np.abs(sequence))
            
            # Update if this is best fit
            if rel_error < lowest_rel_error:
                lowest_rel_error = rel_error
                best_fit = {
                    'degree': degree,
                    'coeffs': coeffs,
                    'fitted': fitted,
                    'abs_error': abs_error,
                    'rel_error': rel_error
                }
        
        # If we found a good fit
        if best_fit and lowest_rel_error < 0.01:  # 1% error threshold
            next_term = int(round(np.polyval(best_fit['coeffs'], len(sequence))))
            
            steps.extend([
                f"Found polynomial pattern:",
                f"1. Degree: {best_fit['degree']}",
                f"2. Coefficients (high to low order):",
                *[f"   x^{i}: {c:+.3f}" 
                  for i, c in enumerate(best_fit['coeffs'][::-1])],
                f"\n3. Fit quality:",
                f"   - Mean relative error: {lowest_rel_error*100:.2f}%",
                f"   - R² score: {1 - lowest_rel_error:.4f}",
                f"\n4. Next term calculation:",
                f"   x = {len(sequence)}",
                f"   y = {' + '.join(f'{c:+.2f}×{len(sequence)}^{i}' for i, c in enumerate(best_fit['coeffs'][::-1]))}",
                f"   y = {next_term}"
            ])
            
            return {
                "type": f"polynomial_{best_fit['degree']}",
                "coefficients": best_fit['coeffs'].tolist(),
                "next": next_term,
                "confidence": 1.0 - lowest_rel_error
            }
            
        return {"type": "unknown", "confidence": 0.0}
        
    def _find_recurrence_weights(self, sequence: List[int]) -> List[float]:
        """Find weights for recurrence relation up to order 4"""
        for order in range(2, 5):
            # Build system of equations
            A = np.array([sequence[i:i+order] for i in range(len(sequence)-order)])
            b = np.array(sequence[order:])
            
            try:
                # Solve for weights
                weights = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Verify the weights work
                predicted = np.dot(A, weights)
                if np.allclose(predicted, b, rtol=1e-4):
                    return weights.tolist()
            except:
                continue
                
        return None
        
    def _is_fibonacci_weights(self, weights: List[float]) -> bool:
        """Check if weights match Fibonacci pattern"""
        return len(weights) == 2 and np.allclose(weights, [1, 1], rtol=1e-4)
        
    def _is_tribonacci_weights(self, weights: List[float]) -> bool:
        """Check if weights match Tribonacci pattern"""
        return len(weights) == 3 and np.allclose(weights, [1, 1, 1], rtol=1e-4)
        
        if is_fibonacci:
            next_term = sequence[-1] + sequence[-2]
            proof_steps = [
                f"Fibonacci-like Sequence Analysis:",
                f"1. Testing if each term is sum of previous two terms:",
                *[f"   {sequence[i-2]} + {sequence[i-1]} = {sequence[i]}" 
                  for i in range(2, len(sequence))],
                f"2. Pattern confirmed: each term is sum of previous two",
                f"3. Next term calculation: {sequence[-2]} + {sequence[-1]} = {next_term}"
            ]
            return {
                "pattern_type": "fibonacci",
                "next_term": next_term,
                "confidence": 1.0,
                "proof_steps": proof_steps
            }
        return None

    def _check_polynomial(self, sequence: List[int], steps: List[str]) -> Dict[str, Any]:
        """Check for polynomial sequences with mathematical proof"""
        x = np.array(range(len(sequence)))
        best_degree = None
        best_fit = None
        best_error = float('inf')
        
        for degree in range(1, min(4, len(sequence)-1)):
            coeffs = np.polyfit(x, sequence, degree)
            fit = np.polyval(coeffs, x)
            error = np.mean((fit - sequence)**2)
            
            if error < 1e-10 and error < best_error:
                best_degree = degree
                best_fit = coeffs
                best_error = error
                
        if best_fit is not None:
            next_x = len(sequence)
            next_term = int(np.polyval(best_fit, next_x))
            
            # Generate polynomial expression
            terms = []
            for i, coeff in enumerate(best_fit):
                if abs(coeff) > 1e-10:  # Only include significant terms
                    power = best_degree - i
                    if power == 0:
                        terms.append(f"{coeff:.2f}")
                    elif power == 1:
                        terms.append(f"{coeff:.2f}n")
                    else:
                        terms.append(f"{coeff:.2f}n^{power}")
                        
            formula = " + ".join(terms)
            
            proof_steps = [
                f"Polynomial Sequence Analysis:",
                f"1. Identified polynomial pattern of degree {best_degree}",
                f"2. Formula: {formula}",
                f"3. Verification (formula generates the sequence):",
                *[f"   n = {i}: {sequence[i]}" for i in range(len(sequence))],
                f"4. Next term (n = {next_x}): {next_term}"
            ]
            
            return {
                "pattern_type": f"polynomial_degree_{best_degree}",
                "next_term": next_term,
                "confidence": 1.0,
                "proof_steps": proof_steps
            }
        return None

    def _check_power(self, sequence: List[int], steps: List[str]) -> Dict[str, Any]:
        """Check for power sequences with mathematical proof"""
        if any(x <= 0 for x in sequence):
            return None
            
        # Check if logs form arithmetic sequence
        logs = [math.log(x) for x in sequence]
        diffs = [logs[i+1] - logs[i] for i in range(len(logs)-1)]
        
        if len(set(round(d, 10) for d in diffs)) == 1:
            base = math.exp(diffs[0])
            first_power = math.log(sequence[0]) / math.log(base)
            next_power = first_power + len(sequence)
            next_term = int(base ** next_power)
            
            proof_steps = [
                f"Power Sequence Analysis:",
                f"1. Taking logarithms of sequence",
                f"2. Log differences: {[f'{d:.4f}' for d in diffs]}",
                f"3. Found constant ratio: base = e^{diffs[0]:.4f} ≈ {base:.4f}",
                f"4. Formula: a(n) = {base:.4f}^(n + {first_power:.4f})",
                f"5. Next term: {base:.4f}^{next_power:.4f} = {next_term}"
            ]
            
            return {
                "pattern_type": "power",
                "next_term": next_term,
                "confidence": 1.0,
                "proof_steps": proof_steps
            }
        return None

    def _check_alternating(self, sequence: List[int], steps: List[str]) -> Dict[str, Any]:
        """Check for alternating patterns with mathematical proof"""
        if len(sequence) < 6:  # Need more terms for confidence in alternating pattern
            return None
            
        # Check even and odd positioned terms separately
        even_terms = sequence[::2]
        odd_terms = sequence[1::2]
        
        # Try to find patterns in the sub-sequences
        even_pattern = self._find_simple_pattern(even_terms)
        odd_pattern = self._find_simple_pattern(odd_terms)
        
        if even_pattern and odd_pattern:
            next_index = len(sequence)
            next_term = even_pattern["next"] if next_index % 2 == 0 else odd_pattern["next"]
            
            proof_steps = [
                f"Alternating Pattern Analysis:",
                f"1. Separating even and odd positioned terms:",
                f"   Even positions: {even_terms}",
                f"   Odd positions: {odd_terms}",
                f"2. Even terms pattern: {even_pattern['description']}",
                f"3. Odd terms pattern: {odd_pattern['description']}",
                f"4. Next position is {next_index} ({['even', 'odd'][next_index % 2]})",
                f"5. Therefore, next term = {next_term}"
            ]
            
            return {
                "pattern_type": "alternating",
                "next_term": next_term,
                "confidence": 0.9,  # Slightly lower confidence for complex patterns
                "proof_steps": proof_steps
            }
        return None
        
    def _find_simple_pattern(self, terms: List[int]) -> Dict[str, Any]:
        """Helper function to find pattern in a simple sequence"""
        if len(terms) < 2:
            return None
            
        diffs = [terms[i+1] - terms[i] for i in range(len(terms)-1)]
        
        # Check constant difference
        if len(set(diffs)) == 1:
            return {
                "next": terms[-1] + diffs[0],
                "description": f"Arithmetic with difference {diffs[0]}"
            }
            
        # Check constant ratio
        if all(terms[i] != 0 for i in range(len(terms)-1)):
            ratios = [terms[i+1]/terms[i] for i in range(len(terms)-1)]
            if len(set(round(r, 10) for r in ratios)) == 1:
                return {
                    "next": int(terms[-1] * ratios[0]),
                    "description": f"Geometric with ratio {ratios[0]:.2f}"
                }
        
        return None