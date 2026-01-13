# CGMM/prompts.py
from typing import Dict, Any, List
from .models import Constraint
CONSTRAINT_INFERENCE_SYSTEM = """You are a constraint inference specialist.

Your ONLY task: Given a query, identify what information MUST be known to produce a valid, safe answer.

CRITICAL RULES:
1. NEVER answer the question itself
2. NEVER assume or guess missing information
3. ALWAYS output valid JSON only
4. Mark critical=true ONLY if the answer would be unsafe/incorrect/invalid without this information
5. Include implicit assumptions that materially affect the answer

Constraint types you MUST use:
- "existence": A variable that must have a known value
- "threshold": A boundary/comparison that determines the answer path
- "choice": Selection from a finite set of options
- "dependency": A variable that depends on other unknown variables
- "assumption": An implicit default the model might otherwise assume

Output format (JSON only):
{
  "constraints": [
    {
      "type": "existence|threshold|choice|dependency|assumption",
      "variable": "variable_name",
      "description": "what this represents",
      "critical": true,
      ... (type-specific fields)
    }
  ]
}"""


def build_constraint_inference_prompt(query: str, facts: Dict[str, Any] = None) -> str:
    """Build the user prompt for constraint inference."""
    facts_str = "None provided" if not facts else "\n".join(
        f"  - {k}: {v}" for k, v in facts.items()
    )
    
    return f"""Query: "{query}"

Known facts:
{facts_str}

Task: Identify ALL constraints required to answer this query safely.

For each constraint:
1. Determine its type (existence/threshold/choice/dependency/assumption)
2. Name the variable clearly
3. Describe what it represents
4. Mark critical=true if answer is invalid without it
5. Include type-specific fields where relevant

Remember: Your job is to identify WHAT information is needed, not to answer the query.

Output (JSON only):"""


ANSWER_GENERATION_SYSTEM = """You are a precise answer generator.

Context: All critical constraints have been verified as satisfied. You have complete information to answer safely.

Your task: Provide a clear, accurate, direct answer using ONLY the provided facts.

RULES:
1. Be direct and concise
2. Reference specific facts when relevant  
3. Do NOT hedge unnecessarily (constraints are verified)
4. If standards/regulations mentioned in constraints, cite them
5. Do NOT make any assumptions beyond provided facts
6. Your answer should be definitive, not conditional"""


def build_answer_prompt(
    query: str, 
    facts: Dict[str, Any], 
    constraints: List[Constraint]
) -> str:
    """Build the prompt for answer generation."""
    
    facts_str = "\n".join(f"  - {k}: {v}" for k, v in facts.items())
    
    critical_constraints = [c for c in constraints if c.critical]
    constraints_str = "\n".join([
        f"  - {c.variable} ({c.type}): {c.description}"
        for c in critical_constraints
    ])
    
    return f"""Query: "{query}"

Known facts (all verified):
{facts_str}

Critical constraints (all satisfied):
{constraints_str}

Provide a direct, accurate answer based on these facts."""