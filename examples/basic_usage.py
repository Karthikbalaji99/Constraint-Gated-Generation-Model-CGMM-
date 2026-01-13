# examples/basic_usage.py

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path so Python can find cgmm module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgmm import CGMM  # Changed from CGGM to CGMM

# Load API key from environment variable
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")


# Initialize CGMM
cgmm = CGMM(api_key=api_key)

# Example 1: Blocked query (missing info)
print("="*60)
print("EXAMPLE 1: Query without sufficient information")
print("="*60)

response1 = cgmm.process(
    query="Should this lease be classified as finance or operating?",
    facts={}
)

print(f"\nStatus: {response1.status}")
if response1.status == "BLOCKED":
    print(f"Reason: {response1.reason}")
    print(f"\nMissing constraints:")
    for c in response1.missing_constraints:
        print(f"  - {c['variable']}: {c['description']}")
    print(f"\nSuggested actions:")
    for action in response1.suggested_actions:
        print(f"  • {action}")

# Example 2: Resume with facts (now allowed)
print("\n" + "="*60)
print("EXAMPLE 2: Resume with missing information")
print("="*60)

response2 = cgmm.resume(
    blocked_response=response1,
    new_facts={
        "lease_term": 84,  # months
        "asset_economic_life": 96,  # months
        "substitution_rights": False,
        "ownership_transfer": False
    }
)

print(f"\nStatus: {response2.status}")
if response2.status == "ANSWER":
    print(f"\nAnswer:\n{response2.answer}")
    print(f"\nConstraints used: {', '.join(response2.constraints_used)}")

# Example 3: Immediate answer (sufficient info provided)
print("\n" + "="*60)
print("EXAMPLE 3: Query with sufficient information upfront")
print("="*60)

response3 = cgmm.process(
    query="Calculate the depreciation for this asset",
    facts={
        "asset_cost": 500000,
        "salvage_value": 50000,
        "useful_life": 10,
        "depreciation_method": "straight_line"
    }
)

print(f"\nStatus: {response3.status}")
if response3.status == "ANSWER":
    print(f"\nAnswer:\n{response3.answer}")