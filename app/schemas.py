"""
Request and response contract for the Claims Risk API.

Pydantic rejects malformed requests before they reach the model (422), and
these models also generate the OpenAPI docs at /docs.
"""

from typing import Literal
from pydantic import BaseModel, Field


class MemberFeatures(BaseModel):
    """One member's features, named exactly as the model's training columns.

    Categories only accept values seen in training, and counts must be
    non-negative, so a caller bug fails loudly instead of being silently scored.
    Rules that span several fields (e.g. medical + pharmacy = total) can't be
    expressed per field; app/main.py checks those.
    """

    age_bucket: Literal["18-29", "30-39", "40-49", "50-59", "60-64"] = Field(
        description="Banded age at enrollment")
    sex: Literal["F", "M"] = Field(description="Recorded sex")
    region: Literal["Midwest", "Northeast", "Southeast", "Southwest", "West"] = Field(
        description="US region")
    total_claims: int = Field(ge=0, description="Approved claims")
    medical_claims: int = Field(ge=0, description="Approved non-pharmacy claims")
    pharmacy_claims: int = Field(ge=0, description="Approved pharmacy claims")
    specialty_drug_fills: int = Field(ge=0, description="Specialty drug fills")
    unique_diagnosis_codes: int = Field(ge=0, description="Distinct diagnosis codes")
    inpatient_claims: int = Field(ge=0, description="Institutional claims")
    has_specialty_drug: bool = Field(description="Any specialty drug fill")


class PredictionResponse(BaseModel):
    """The decision, the probability behind it, and the dollar threshold it refers to.

    threshold_dollars makes the response self-describing: if a second model is
    ever trained at a different threshold, a caller can tell them apart.
    """

    is_high_cost: bool = Field(description="True if probability >= the decision threshold")
    probability: float = Field(ge=0.0, le=1.0, description="Predicted probability of high cost")
    threshold_dollars: int = Field(description="Plan-paid dollar threshold the label is defined at")