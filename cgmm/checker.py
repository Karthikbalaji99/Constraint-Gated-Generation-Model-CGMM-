# CGMM/checker.py

from typing import List, Dict, Any, Tuple
from .models import Constraint, ConstraintType


class SufficiencyChecker:
    """Deterministically checks if constraints are satisfied."""
    
    def check(
        self, 
        constraints: List[Constraint], 
        facts: Dict[str, Any]
    ) -> Tuple[bool, List[Constraint]]:
        """
        Check if all critical constraints are satisfied.
        
        Args:
            constraints: List of all constraints
            facts: Known facts
            
        Returns:
            (is_sufficient, missing_critical_constraints)
        """
        critical = [c for c in constraints if c.critical]
        missing = []
        
        for constraint in critical:
            if not self._is_satisfied(constraint, facts):
                missing.append(constraint)
        
        is_sufficient = len(missing) == 0
        return (is_sufficient, missing)
    
    def _is_satisfied(self, constraint: Constraint, facts: Dict[str, Any]) -> bool:
        """Check if a single constraint is satisfied."""
        
        if constraint.type == ConstraintType.EXISTENCE:
            # Variable must exist and not be None
            return (
                constraint.variable in facts 
                and facts[constraint.variable] is not None
            )
        
        elif constraint.type == ConstraintType.THRESHOLD:
            # Variable must exist to check threshold
            # (The threshold check itself happens in answer generation)
            return constraint.variable in facts
        
        elif constraint.type == ConstraintType.CHOICE:
            # Variable must exist and be in valid options
            if constraint.variable not in facts:
                return False
            if not constraint.options:
                return True  # No options specified, just existence check
            return facts[constraint.variable] in constraint.options
        
        elif constraint.type == ConstraintType.DEPENDENCY:
            # All blocking dependencies must be present
            if not constraint.blocking_dependencies:
                return True  # No hard blocks
            return all(
                dep in facts and facts[dep] is not None
                for dep in constraint.blocking_dependencies
            )
        
        elif constraint.type == ConstraintType.ASSUMPTION:
            # If requires explicit confirmation, must be in facts
            if constraint.requires_explicit_confirmation:
                return constraint.variable in facts
            # Otherwise, implicit assumptions are considered "satisfied"
            return True
        
        return False