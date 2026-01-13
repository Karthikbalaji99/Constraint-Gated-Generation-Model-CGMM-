# CGMM/CGMM.py

from typing import Dict, Any, Union
from openai import OpenAI

from .models import CGMMResponse, BlockedResponse, AnswerResponse
from .inference import ConstraintInferenceEngine
from .checker import SufficiencyChecker
from .gate import GateDecision
from .response import BlockedResponseBuilder, AnswerGenerator


class CGMM:
    """
    Constraint-Gated Generative Model
    
    Main orchestrator that coordinates all modules.
    """
    
    def __init__(
        self, 
        api_key: str,
        model: str = "gpt-4o",
        mode: str = "hard_block"
    ):
        """
        Initialize CGMM.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
            mode: Gating mode (currently only "hard_block" supported)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.mode = mode
        
        # Initialize modules
        self.inference_engine = ConstraintInferenceEngine(self.client, model)
        self.sufficiency_checker = SufficiencyChecker()
        self.gate = GateDecision()
        self.blocked_builder = BlockedResponseBuilder()
        self.answer_generator = AnswerGenerator(self.client, model)
    
    def process(
        self, 
        query: str, 
        facts: Dict[str, Any] = None
    ) -> CGMMResponse:
        """
        Main entry point: Process a query with optional facts.
        
        Args:
            query: The user's question
            facts: Known facts (optional)
            
        Returns:
            Either BlockedResponse or AnswerResponse
        """
        facts = facts or {}
        
        print(f"\n🔍 Processing query: {query}")
        print(f"📋 Known facts: {list(facts.keys()) if facts else 'None'}")
        
        # Step 1: Infer constraints
        print("\n⚙️  Step 1: Inferring constraints...")
        constraints = self.inference_engine.infer(query, facts)
        print(f"   Found {len(constraints)} constraints ({sum(c.critical for c in constraints)} critical)")
        
        # Step 2: Check sufficiency
        print("\n⚙️  Step 2: Checking sufficiency...")
        sufficient, missing = self.sufficiency_checker.check(constraints, facts)
        
        if sufficient:
            print(f"   ✅ All critical constraints satisfied")
        else:
            print(f"   ❌ Missing {len(missing)} critical constraints")
        
        # Step 3: Gate decision
        print("\n⚙️  Step 3: Gate decision...")
        decision = self.gate.decide(sufficient, missing)
        print(f"   🚦 Decision: {decision}")
        
        # Step 4: Generate output
        if decision == "BLOCK":
            print("\n🚫 BLOCKED - Generating structured response")
            return self.blocked_builder.build(query, missing, facts)
        else:
            print("\n✅ ALLOWED - Generating answer")
            return self.answer_generator.generate(query, facts, constraints)
    
    def resume(
        self, 
        blocked_response: Union[BlockedResponse, Dict], 
        new_facts: Dict[str, Any]
    ) -> CGMMResponse:
        """
        Resume processing after receiving missing information.
        
        Args:
            blocked_response: Previous BlockedResponse
            new_facts: New facts to add
            
        Returns:
            New CGMMResponse (might still be blocked if insufficient)
        """
        # Handle both BlockedResponse object and dict
        if isinstance(blocked_response, dict):
            query = blocked_response["query_original"]
            old_facts = blocked_response.get("facts_original", {})
        else:
            query = blocked_response.query_original
            old_facts = blocked_response.facts_original
        
        # Merge facts
        all_facts = {**old_facts, **new_facts}
        
        print(f"\n🔄 Resuming query with {len(new_facts)} new facts")
        
        # Re-process
        return self.process(query, all_facts)