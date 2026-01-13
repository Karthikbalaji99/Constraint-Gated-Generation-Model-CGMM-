# CGMM/models.py

from enum import Enum
from typing import Optional, List, Union, Any, Dict
from pydantic import BaseModel, Field


class ConstraintType(str, Enum):
    """The five constraint types in our taxonomy."""
    EXISTENCE = "existence"
    THRESHOLD = "threshold"
    CHOICE = "choice"
    DEPENDENCY = "dependency"
    ASSUMPTION = "assumption"


class Constraint(BaseModel):
    """Represents a single constraint required to answer a query."""
    
    # Core fields (all constraints have these)
    type: ConstraintType
    variable: str
    description: str
    critical: bool = True
    
    # Type-specific fields (EXISTENCE)
    domain: Optional[str] = None  # "integer", "string", "date", "boolean"
    unit: Optional[str] = None    # "months", "INR", "kg"
    
    # Type-specific fields (THRESHOLD)
    operator: Optional[str] = None      # ">=", "<", "==", etc.
    threshold: Optional[float] = None
    
    # Type-specific fields (CHOICE)
    options: Optional[List[str]] = None
    mutually_exclusive: Optional[bool] = True
    
    # Type-specific fields (DEPENDENCY)
    depends_on: Optional[List[str]] = None
    formula: Optional[str] = None
    blocking_dependencies: Optional[List[str]] = None
    
    # Type-specific fields (ASSUMPTION)
    default_assumed: Optional[str] = None
    alternatives: Optional[List[str]] = None
    requires_explicit_confirmation: Optional[bool] = None
    
    # Metadata
    reference: Optional[str] = None  # Standard/law reference
    materiality: Optional[str] = None  # "low", "medium", "high"


class BlockedResponse(BaseModel):
    """Response when query is blocked due to missing constraints."""
    status: str = "BLOCKED"
    reason: str
    missing_constraints: List[Dict[str, Any]]
    explanation: str
    suggested_actions: List[str]
    query_original: str
    facts_original: Dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    """Response when query can be answered."""
    status: str = "ANSWER"
    answer: str
    constraints_used: List[str]
    facts_applied: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Type alias for response
CGMMResponse = Union[BlockedResponse, AnswerResponse]