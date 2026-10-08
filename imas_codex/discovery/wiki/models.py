"""
Data models for wiki discovery pipeline.

These models define runtime structures for the LLM-based wiki scoring phase.
Mirrors the structure of discovery/paths/models.py for consistency.

The Pydantic models (WikiScoreResult, WikiScoreBatch) are used for LLM
structured output - LiteLLM parses responses directly into these.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.discovery.base.scoring import (
    ContentScoreFields,
    purpose_weighted_composite,
)
from imas_codex.graph.models import ContentPurpose

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


class ImageScoreResult(ContentScoreFields):
    """Vision result for callers that still use shared image scoring."""

    id: str = Field(description="The image ID (echo from input)")

    mermaid_diagram: str = Field(
        default="",
        description="Mermaid diagram representing the structure of schematics, "
        "block diagrams, or data flow images. Use graph LR or graph TD syntax. "
        "Empty string for non-schematic images (plots, photos, etc.).",
    )

    ocr_text: str = Field(
        default="",
        description="All visible text in the image: axis labels, legends, titles, "
        "MDSplus paths, parameter values. Empty string if no text visible.",
    )

    description: str = Field(
        description="Detailed physics-aware description of image content. "
        "Include specific quantities, diagnostics, tree paths, conventions. "
        "Describe what the image shows in fusion physics terms, not visual appearance. "
        "Length scales with content richness: 1-2 sentences for simple images, "
        "full paragraph for complex schematics or multi-panel plots."
    )

    purpose: ContentPurpose
    reasoning: str = ""
    keywords: list[str] = Field(default_factory=list)
    physics_domain: PhysicsDomain = PhysicsDomain.GENERAL
    should_ingest: bool
    skip_reason: str = ""


class ImageCaptionResult(BaseModel):
    """Caption, OCR, and diagram text requested by the wiki image worker."""

    id: str
    description: str
    ocr_text: str = ""
    mermaid_diagram: str = ""


class ImageCaptionBatch(BaseModel):
    """Caption-only vision responses in input order."""

    results: list[ImageCaptionResult]


class ImageScoreBatch(BaseModel):
    """Batch of image scoring results from VLM.

    This is the top-level model passed to LiteLLM's response_format.
    """

    results: list[ImageScoreResult] = Field(
        description="List of scoring results, one per input image, in order"
    )


def grounded_image_score(
    scores: dict[str, float],
    purpose: ContentPurpose,
) -> float:
    """Compute combined score for an image.

    Delegates to the shared ``purpose_weighted_composite`` function.
    """
    return purpose_weighted_composite(scores, purpose)
