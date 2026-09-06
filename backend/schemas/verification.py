"""
Verification Schemas — Models for the Reliability & Verification Framework.

Phase 7 component.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str = Field(description="The falsifiable claim extracted from the answer.")
    is_supported: bool = Field(description="True if the claim is fully supported by the evidence.")
    reasoning: str = Field(description="Explanation of why it is supported or unsupported.")


class Contradiction(BaseModel):
    description: str = Field(description="Description of the conflicting evidence.")
    source_a: str = Field(description="Source containing the first claim.")
    source_b: str = Field(description="Source containing the conflicting claim.")


class VerificationResult(BaseModel):
    is_fully_supported: bool = Field(description="True if ALL claims are supported and NO unaddressed contradictions exist.")
    claims: List[Claim] = Field(default_factory=list, description="List of all extracted claims and their support status.")
    contradictions: List[Contradiction] = Field(default_factory=list, description="List of detected contradictions across sources.")
    revised_answer: Optional[str] = Field(default=None, description="The original answer appended with warning blocks if necessary.")
