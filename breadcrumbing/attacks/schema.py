# attacker/schema.py

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List

# =========================
# Decision
# planner → hook_search
# =========================


@dataclass
class Payload:
    # --- search related ---
    title: Optional[str] = None
    link: Optional[str] = None
    target_urls: Optional[List[str]] = None
    snippet: Optional[str] = None
    position: Optional[int] = None
    date: Optional[str] = None
    source: Optional[str] = None

    # --- visit related ---
    page_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    attack: bool
    payload: Optional[Payload]
    reflection: str

    # ---- LLM / JSON 兼容 ----
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Decision":
        payload = d.get("payload")
        return cls(
            attack=d["attack"],
            payload=Payload(**payload) if payload else None,
            reflection=d.get("reflection", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "attack": self.attack,
            "reflection": self.reflection,
        }
        if self.payload:
            result["payload"] = self.payload.to_dict()
        return result


@dataclass
class Event:
    id: int
    event_type: str  # "search" | "visit"

    input: dict  # {"query": "..."} or {"url": "..."}
    raw: Any  # tool 原始结果

    decision: dict  # planner decision (可直接放 Decision.to_dict())
    output: Any  # hook 最终返回结果

    timestamp: float
    behavior_meta: dict = field(default_factory=dict)  # 行为日志


@dataclass
class Strategy:
    """
    Canonical strategy schema.

    YAML schema:

    - name: str
      description: str
      goal: str
      steps:
        - search: str
        - visit: str
      usage_conditions: List[str]   # When this strategy is likely to succeed
      avoid_conditions: List[str]    # When this strategy should not be used
    """

    name: str
    description: str
    goal: str
    steps: List[Dict[str, str]] = field(default_factory=list)

    # Strategy applicability conditions (for attacker selection and reflection analysis)
    usage_conditions: List[str] = field(default_factory=list)
    avoid_conditions: List[str] = field(default_factory=list)

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Strategy.name is required")

        if not isinstance(self.steps, list) or len(self.steps) == 0:
            raise ValueError(f"{self.name}: steps must be non-empty list")

        for step in self.steps:
            if not isinstance(step, dict):
                raise ValueError(f"{self.name}: step must be dict")

            if not any(k in step for k in ("search", "visit")):
                raise ValueError(
                    f"{self.name}: each step must contain 'search' or 'visit'"
                )

    # -----------------------------------------------------
    # Formatting (for prompt / planner)
    # -----------------------------------------------------

    def to_prompt_text(self) -> str:
        """
        Convert strategy into LLM-friendly text.
        """
        lines = [
            f"Strategy: {self.name}",
            f"Description: {self.description}",
            f"Goal: {self.goal}",
            "Steps:",
        ]

        for step in self.steps:
            if "search" in step:
                lines.append(f"  - search: {step['search']}")
            if "visit" in step:
                lines.append(f"  - visit: {step['visit']}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "steps": self.steps,
            "usage_conditions": self.usage_conditions,
            "avoid_conditions": self.avoid_conditions,
        }
