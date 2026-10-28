"""Attacks included in the paper's comparisons."""

from .aitm import AiTMSearchAttacker
from .mast import MASTDirectSearchAttacker
from .intent_hijacking import IntentHijackingSearchAttacker
from .evo_attacker import EvoAttacker
from .base import BaseAttacker as BaseAttacker
from .noop import NoOpAttacker
from .ach import ACHAttacker
from .agentdojo import AgentDojoAttacker
from .static_inject import GeoAttacker
from .safesearch import SafeSearchAttacker
from .objective_drift import ObjectiveDriftAttacker


def build_attacker(name: str, **kwargs):
    """Construct one paper method; case_context supplies the complete task row."""
    attackers = {
        "aitm": AiTMSearchAttacker,
        "mast": MASTDirectSearchAttacker,
        "intent_hijacking": IntentHijackingSearchAttacker,
        "evo_attacker": EvoAttacker,
        "no_attack": NoOpAttacker,
        "ach": ACHAttacker,
        "agentdojo": AgentDojoAttacker,
        "geo": GeoAttacker,
        "safesearch": SafeSearchAttacker,
        "objective_drift": ObjectiveDriftAttacker,
    }
    if name.lower() not in attackers:
        raise ValueError(f"Unknown attacker {name!r}; choose {', '.join(attackers)}")
    if name.lower() == "no_attack":
        return NoOpAttacker()
    return attackers[name.lower()](**kwargs)
