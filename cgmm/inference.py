# CGMM/inference.py

import json
from typing import List, Dict, Any
from openai import OpenAI

from .models import Constraint
from .prompts import (
    CONSTRAINT_INFERENCE_SYSTEM,
    build_constraint_inference_prompt
)


class ConstraintInferenceEngine:
    """Extracts constraints from queries using LLM."""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
    
    def infer(self, query: str, facts: Dict[str, Any] = None) -> List[Constraint]:
        """
        Infer constraints from a query.
        
        Args:
            query: The user's question
            facts: Known facts (optional)
            
        Returns:
            List of Constraint objects
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CONSTRAINT_INFERENCE_SYSTEM},
                    {"role": "user", "content": build_constraint_inference_prompt(query, facts)}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # Low temperature for consistency
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Parse into Constraint objects
            # Parse into Constraint objects
            constraints = []
            for c_dict in result.get("constraints", []):
                try:
                    # Clean up percentage thresholds
                    if c_dict.get("threshold") and isinstance(c_dict["threshold"], str):
                        # Convert "75%" to 0.75
                        if "%" in c_dict["threshold"]:
                            c_dict["threshold"] = float(c_dict["threshold"].replace("%", "")) / 100
                    
                    constraint = Constraint(**c_dict)
                    constraints.append(constraint)
                except Exception as e:
                    print(f"Warning: Failed to parse constraint {c_dict}: {e}")
                    continue
            
            return constraints
            
        except Exception as e:
            print(f"Error in constraint inference: {e}")
            raise