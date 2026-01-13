# CGMM/gate.py

from typing import List
from .models import Constraint


class GateDecision:
    """Makes the BLOCK/ALLOW decision."""
    
    @staticmethod
    def decide(sufficient: bool, missing: List[Constraint]) -> str:
        """
        Hard block mode: BLOCK if any critical constraint missing.
        
        Returns:
            "ALLOW" or "BLOCK"
        """
        return "ALLOW" if sufficient else "BLOCK"