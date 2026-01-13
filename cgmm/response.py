# CGMM/response.py

from typing import List, Dict, Any
from datetime import datetime
from openai import OpenAI

from .models import Constraint, BlockedResponse, AnswerResponse
from .prompts import ANSWER_GENERATION_SYSTEM, build_answer_prompt


class BlockedResponseBuilder:
    """Builds structured response when blocked."""
    
    def build(
        self, 
        query: str, 
        missing: List[Constraint],
        facts: Dict[str, Any] = None
    ) -> BlockedResponse:
        """Create a blocked response with missing constraints."""
        
        facts = facts or {}
        
        missing_dicts = [
            {
                "variable": c.variable,
                "type": c.type.value,
                "description": c.description,
                "critical": c.critical,
                "reference": c.reference,
                **({"options": c.options} if c.options else {}),
                **({"unit": c.unit} if c.unit else {})
            }
            for c in missing
        ]
        
        explanation = self._generate_explanation(query, missing)
        actions = self._suggest_actions(missing)
        
        return BlockedResponse(
            reason="Insufficient information to produce a valid answer",
            missing_constraints=missing_dicts,
            explanation=explanation,
            suggested_actions=actions,
            query_original=query,
            facts_original=facts
        )
    
    def _generate_explanation(self, query: str, missing: List[Constraint]) -> str:
        """Generate human-readable explanation."""
        variables = [c.variable for c in missing]
        var_list = ", ".join(variables)
        
        return (
            f"Cannot safely answer '{query}' because the following "
            f"critical information is missing: {var_list}. "
            f"Please provide these values to proceed."
        )
    
    def _suggest_actions(self, missing: List[Constraint]) -> List[str]:
        """Generate actionable next steps."""
        actions = []
        
        for c in missing:
            if c.type == "choice" and c.options:
                actions.append(f"Select {c.variable} from: {', '.join(c.options)}")
            elif c.type == "assumption":
                actions.append(
                    f"Confirm or reject assumption for {c.variable}: {c.default_assumed}"
                )
            elif c.unit:
                actions.append(f"Provide {c.variable} (in {c.unit})")
            else:
                actions.append(f"Provide value for {c.variable}")
        
        return actions


class AnswerGenerator:
    """Generates answers when gate allows."""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
    
    def generate(
        self, 
        query: str, 
        facts: Dict[str, Any],
        constraints: List[Constraint]
    ) -> AnswerResponse:
        """Generate answer with verified constraints."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANSWER_GENERATION_SYSTEM},
                {"role": "user", "content": build_answer_prompt(query, facts, constraints)}
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        return AnswerResponse(
            answer=answer,
            constraints_used=[c.variable for c in constraints if c.critical],
            facts_applied=list(facts.keys()),
            metadata={
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
        )