"""
Strategy Manager

Responsibilities
----------------
- Load strategies from YAML files
- Validate schema
- Maintain in-memory registry
- Provide formatted prompt text for planner / LLM

This module is DATA MANAGEMENT ONLY.
No attack logic or planning logic here.
"""

from __future__ import annotations

import os
import yaml
from typing import Dict, List, Optional
from ..schema import Strategy

# =========================================================
# Strategy Manager
# =========================================================


class StrategyManager:
    """
    Registry + Loader + Saver
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        strategies_path: Optional[str] = None,
    ):
        """
        Args:
            model_name: Model name (used for learned/ save path, not loading).
            strategies_path: If provided, load a single strategies YAML file directly.
        """

        self.model_name = model_name
        self.strategies_path = strategies_path
        self._strategies: Dict[str, Strategy] = {}

        if strategies_path:
            self._load_yaml_if_exists(strategies_path)
        else:
            self._load_default()

    # =====================================================
    # Loading
    # =====================================================

    def _load_default(self):
        # Strategy selection is explicit in the release presets.
        return

    def _load_yaml_if_exists(self, path: str):
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []

        for item in data:
            strategy = Strategy(**item)
            strategy.validate()
            self._strategies[strategy.name] = strategy

    # =====================================================
    # CRUD
    # =====================================================

    def add_strategy(self, strategy: Strategy, save: bool = False):
        strategy.validate()
        if strategy.name in self._strategies:
            raise ValueError(f"Duplicate strategy name: {strategy.name}")
        self._strategies[strategy.name] = strategy
        if save:
            self.save_strategy(strategy)

    def get_strategy(self, name: str) -> Strategy:
        return self._strategies[name]

    def list_strategies(self) -> List[str]:
        return sorted(self._strategies.keys())

    def all(self) -> List[Strategy]:
        return list(self._strategies.values())

    # =====================================================
    # Formatting for Planner
    # =====================================================

    def format_all_for_prompt(self) -> str:
        texts = []
        for s in self.all():
            texts.append(f"Strategy: {s.name}")
            texts.append(f"Description: {s.description}")
            texts.append(f"Goal: {s.goal}")
            texts.append("Steps:")
            for step in s.steps:
                if "search" in step:
                    texts.append(f"  - search: {step['search']}")
                if "visit" in step:
                    texts.append(f"  - visit: {step['visit']}")
            texts.append("")
        return "\n".join(texts).strip()

    # =====================================================
    # Persistence
    # =====================================================

    def save_strategy(self, strategy: Strategy):
        if not self.model_name:
            raise ValueError("model_name required to save learned strategy")
        save_dir = os.getenv("BREADCRUMBING_STRATEGY_OUTPUT", "outputs/strategies")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir,
            f"{self.model_name.replace(chr(47), chr(95)).replace(chr(58), chr(95))}.yaml",
        )
        existing = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f) or []
        existing.append(strategy.to_dict())
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(existing, f, sort_keys=False, allow_unicode=True)
