from typing import Dict, List, Any
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .base_agent import BaseAgent

class Decomposer(BaseAgent):
    """Agent responsible for breaking down complex problems into manageable steps"""
    
    def __init__(self, model_name: str = "t5-small"):
        super().__init__()
        self.model_name = model_name
        self._load_model()
        
    def _load_model(self):
        """Load the T5 model and tokenizer"""
        self._log_step(f"Loading model {self.model_name}")
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Break down a problem into sequential steps
        
        Args:
            input_data (Dict[str, Any]): Contains the question to decompose
            
        Returns:
            Dict[str, Any]: Decomposed steps and metadata
        """
        question = input_data["question"]
        self._log_step("Processing question", {"question": question})
        
        # Generate decomposition prompt
        prompt = self._create_prompt(question)
        
        # Generate steps using T5
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512)
            outputs = self.model.generate(**inputs)
            steps_text = self.tokenizer.decode(outputs[0])
            steps = self._clean_steps(steps_text)
        except Exception as e:
            self.logger.error(f"Failed to generate steps: {str(e)}")
            return {"error": str(e)}
        
        result = {
            "original_question": question,
            "steps": steps,
            "metadata": {
                "model_used": self.model_name,
                "num_steps": len(steps)
            }
        }
        
        self._log_step("Decomposition complete", {"num_steps": len(steps)})
        return result
    
    def _create_prompt(self, question: str) -> str:
        """Create a prompt for step generation"""
        return (
            "Break this problem into clear, logical steps:\n\n"
            f"Problem: {question}\n\n"
            "Steps:"
        )
    
    def _clean_steps(self, steps: str) -> List[str]:
        """Clean and format the generated steps"""
        # Split into lines
        lines = steps.split("\n")
        
        # Clean each line
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove step numbers if present
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Remove any special tokens
            line = re.sub(r'<[^>]+>', '', line)
            
            if line:
                cleaned.append(line)
        
        return cleaned