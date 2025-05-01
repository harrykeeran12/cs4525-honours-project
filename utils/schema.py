from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class ErrorType(Enum):
    """Strict definition of the multiple errors that can occur."""

    TranscriptionError = "Transcription Error"
    InternalInconsistency = "Internal Inconsistency"
    Omission = "Omission"
    ExtraneousStatement = "Extraneous Statement"


class RadiologyError(BaseModel):
    """This class serves as a schema to act as structured output for the models."""

    errorType: ErrorType
    errorPhrases: list[str]
    errorExplanation: list[str]


class RadiologyErrors(BaseModel):
    """Adding multiple errors for structured output."""

    errorsForWholeText: List[RadiologyError] | None
