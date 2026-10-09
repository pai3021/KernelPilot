"""Optional, evidence-gated harness-evolution support for KernelPilot."""

from .config import SelfEvolutionConfig, load_config
from .coordinator import EvolutionCoordinator
from .models import HarnessProposal, RetrospectiveContext

__all__ = [
    "EvolutionCoordinator",
    "HarnessProposal",
    "RetrospectiveContext",
    "SelfEvolutionConfig",
    "load_config",
]
