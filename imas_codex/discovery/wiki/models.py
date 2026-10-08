"""
Data models for wiki discovery pipeline.

These models hold language-model descriptions and vision captions.

The Pydantic models (WikiScoreResult, WikiScoreBatch) are used for LLM
structured output - LiteLLM parses responses directly into these.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ============================================================================
# Pydantic Models for LLM Structured Output
# ============================================================================


class WikiScoreResult(BaseModel):
    """The text description retained from the language model."""

    id: str = Field(description="The page ID (echo from input)")

    description: str = Field(
        description="Concise description of page contents (1-2 sentences)"
    )


class WikiScoreBatch(BaseModel):
    """Batch of wiki page scoring results from LLM.

    This is the top-level model passed to LiteLLM's response_format.
    The LLM returns a list of WikiScoreResult objects.
    """

    results: list[WikiScoreResult] = Field(
        description="List of scoring results, one per input page, in order"
    )


# ============================================================================
# Document Scoring Pydantic Models (LLM Structured Output)
# ============================================================================


class DocumentScoreResult(BaseModel):
    """The text description retained from the language model."""

    id: str = Field(description="The document ID (echo from input)")

    description: str = Field(
        description="Concise description of document contents (1-2 sentences)"
    )


class DocumentScoreBatch(BaseModel):
    """Batch of document scoring results from LLM.

    This is the top-level model passed to LiteLLM's response_format.
    """

    results: list[DocumentScoreResult] = Field(
        description="List of scoring results, one per input document, in order"
    )


# ============================================================================
# Image Scoring Pydantic Models (VLM Structured Output)
# ============================================================================


class ImageCaptionResult(BaseModel):
    """Caption, OCR, and diagram text requested by the shared image worker."""

    id: str
    description: str
    ocr_text: str = ""
    mermaid_diagram: str = ""


class ImageCaptionBatch(BaseModel):
    """Caption-only vision responses in input order."""

    results: list[ImageCaptionResult]
