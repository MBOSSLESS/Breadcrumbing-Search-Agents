"""Trace-Guided Strategy Evolution (TGSE)."""

from .config import TGSEConfig

__all__ = ["TGSEConfig", "evolve", "evaluate_selected"]


def evolve(*args, **kwargs):
    from .search import evolve as run

    return run(*args, **kwargs)


def evaluate_selected(*args, **kwargs):
    from .search import evaluate_selected as run

    return run(*args, **kwargs)
