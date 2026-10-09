"""Small, explicit feature configuration; advanced evolution defaults on."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SelfEvolutionConfig:
    enabled: bool = True
    runtime: str = "codex"
    after_campaign: bool = True
    max_proposals: int = 1

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None = None) -> "SelfEvolutionConfig":
        data = dict(value or {})
        config = cls(
            enabled=bool(data.get("enabled", True)),
            runtime=str(data.get("runtime", "codex")),
            after_campaign=bool(data.get("after_campaign", True)),
            max_proposals=int(data.get("max_proposals", 1)),
        )
        if config.max_proposals != 1:
            raise ValueError("V1 self-evolution allows exactly one proposal per retrospective")
        if not config.runtime.strip():
            raise ValueError("self_evolution.runtime must not be empty")
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "runtime": self.runtime,
            "after_campaign": self.after_campaign,
            "max_proposals": self.max_proposals,
        }


def load_config(path: Path | None = None) -> SelfEvolutionConfig:
    """Load an optional TOML override, retaining the documented ON default."""
    if path is None:
        return SelfEvolutionConfig()
    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10 WSL control plane
            import tomli as tomllib
        raw = tomllib.loads(Path(path).read_text())
    except FileNotFoundError:
        return SelfEvolutionConfig()
    section = raw.get("self_evolution", {})
    if not isinstance(section, Mapping):
        raise ValueError("[self_evolution] must be a TOML table")
    return SelfEvolutionConfig.from_dict(section)
