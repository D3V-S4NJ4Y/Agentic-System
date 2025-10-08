from abc import ABC, abstractmethod
from typing import Dict, Any, List
import re
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache = {}
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and return results
        
        Args:
            input_data (Dict[str, Any]): Input data to process
            
        Returns:
            Dict[str, Any]: Processed results
        """
        pass
    
    def _log_step(self, message: str):
        """Log a step in the agent's process"""
        self.logger.info(f"[{self.__class__.__name__}] {message}")
    
    @lru_cache(maxsize=1000)
    def _get_cached_result(self, problem_hash: str) -> Dict[str, Any]:
        """Get cached result for a problem"""
        return self._cache.get(problem_hash)
    
    def _cache_result(self, problem_hash: str, result: Dict[str, Any]):
        """Cache a result for future use"""
        self._cache[problem_hash] = result
    
    def _generate_problem_hash(self, problem: Dict[str, Any]) -> str:
        """Generate a unique hash for a problem"""
        return json.dumps(problem, sort_keys=True)
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        return [float(n) for n in re.findall(r'-?\d*\.?\d+', text)]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        keywords = []
        indicators = {
            'math': r'calculate|compute|solve|equation|value',
            'logic': r'deduce|infer|reason|conclude',
            'sequence': r'pattern|series|next|follow',
            'probability': r'chance|likely|probability|odds',
            'optimization': r'maximize|minimize|optimal|best'
        }
        
        for category, pattern in indicators.items():
            if re.search(pattern, text, re.IGNORECASE):
                keywords.append(category)
        
        return keywords
    
    def _identify_constraints(self, text: str) -> List[str]:
        """Identify constraints in the problem"""
        constraints = []
        constraint_patterns = [
            r'must\s+\w+',
            r'cannot\s+\w+',
            r'only\s+\w+',
            r'at\s+least\s+\w+',
            r'at\s+most\s+\w+',
            r'exactly\s+\w+'
        ]
        
        for pattern in constraint_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            constraints.extend(match.group() for match in matches)
        
        return constraints
    
    def _extract_logical_relations(self, text: str) -> List[Dict[str, str]]:
        """Extract logical relations from text"""
        relations = []
        patterns = {
            'if_then': r'if\s+(.+?)\s+then\s+(.+?)(?:\.|$)',
            'because': r'(.+?)\s+because\s+(.+?)(?:\.|$)',
            'therefore': r'(.+?)\s+therefore\s+(.+?)(?:\.|$)'
        }
        
        for rel_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relations.append({
                    'type': rel_type,
                    'antecedent': match.group(1).strip(),
                    'consequent': match.group(2).strip()
                })
        
        return relations
    
    def _verify_logical_consistency(self, statements: List[Dict[str, str]]) -> bool:
        """Verify logical consistency of extracted statements"""
        seen_statements = set()
        contradictions = False
        
        for statement in statements:
            key = f"{statement['antecedent']}_{statement['consequent']}"
            if key in seen_statements:
                continue
                
            neg_key = f"{statement['consequent']}_{statement['antecedent']}"
            if neg_key in seen_statements:
                contradictions = True
                break
                
            seen_statements.add(key)
        
        return not contradictions
        msg = f"[{self.__class__.__name__}] {step}"
        if details:
            msg += f": {details}"
        self.logger.info(msg)