"""
Schemas for the single-strategy TGSE pipeline.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class StrategyUpdate(BaseModel):
    """Complete replacement strategy returned by reflection."""

    name: str
    description: str
    goal: str
    steps: List[Dict[str, str]]
    usage_conditions: List[str] = Field(default_factory=list)
    avoid_conditions: List[str] = Field(default_factory=list)


class CaseDiagnostic(BaseModel):
    """Case-level diagnosis to preserve explainability."""

    case_id: str
    bucket: Optional[str] = None
    pattern: Optional[str] = None
    diagnosis: str
    lessons: List[str] = Field(default_factory=list)


class Analysis(BaseModel):
    """Main reflection analysis block, aligned with the main pipeline style."""

    successful_patterns: List[str] = Field(default_factory=list)
    failure_patterns: List[str] = Field(default_factory=list)
    root_causes: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)


class StrategyDiff(BaseModel):
    """Natural-language audit trail for strategy changes."""

    summary: str = ""
    added: List[str] = Field(default_factory=list)
    removed: List[str] = Field(default_factory=list)
    modified: List[str] = Field(default_factory=list)
    preserved: List[str] = Field(default_factory=list)
    rollback_notes: List[str] = Field(default_factory=list)


class ReflectionResponse(BaseModel):
    """Structured reflection output for one generation."""

    analysis: Analysis
    self_reasoning: str
    case_diagnostics: List[CaseDiagnostic] = Field(default_factory=list)
    updated_strategy: StrategyUpdate
    expected_impact: str
    additional_notes: str = ""
    change_rationale: List[str] = Field(default_factory=list)
    strategy_diff: StrategyDiff = Field(default_factory=StrategyDiff)


class LessonReflectionResponse(BaseModel):
    """Compact lesson payload stored in rolling lesson memory."""

    summary: str
    lessons: List[str] = Field(default_factory=list)
    case_links: List[str] = Field(default_factory=list)


__all__ = ["ReflectionResponse", "LessonReflectionResponse"]
