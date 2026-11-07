"""The ForceCRM analysis as explicit, cached stages (see docs/ARCHITECTURE.md §2)."""
from .engine import Runner, Stage
from .definitions import STAGES, ORDER

__all__ = ['Runner', 'Stage', 'STAGES', 'ORDER']
