from pydantic import BaseModel


class RadiologyError(BaseModel):
    """This class serves as a schema to act as as structured output for the models."""

    errorType: list[str]
    errorPhrases: list[str]
    errorExplanation: list[str]
