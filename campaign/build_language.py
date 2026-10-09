"""Harness-owned build-language selection for planned branch strategies."""

from __future__ import annotations

import re
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


# These are the Triton strategy_family values emitted by the persisted AKO4X
# planner artifacts.  Keep this an explicit mapping: planner output is an
# unconstrained string, so fuzzy matching could silently select a DSL backend
# for an unrelated strategy.
TRITON_STRATEGY_FAMILIES = frozenset({
    "Triton register-tiled implicit-GEMM",
    "custom fused Triton kernel",
    "custom-triton-batchnorm-fusion",
    "custom_fused_triton_kernel",
    "custom_fused_triton_reduction",
    "custom_triton_batchnorm",
    "custom_triton_diagonal_kernel",
    "custom_triton_elementwise",
    "custom_triton_fused_kernel",
    "custom_triton_fused_reduction",
    "custom_triton_fusion",
    "custom_triton_gemm",
    "custom_triton_kernel",
    "custom_triton_large_k_tiling",
    "custom_triton_online_softmax",
    "custom_triton_pointwise_kernel",
    "custom_triton_pooling",
    "custom_triton_reduction",
    "custom_triton_shape_specialized_gemm",
    "custom_triton_smallk",
    "custom_triton_specialization",
    "custom_triton_splitk",
    "custom_triton_tiled_gemm",
    "custom_triton_tiling",
    "custom_triton_vectorized",
    "custom_triton_warp_reduction",
    "custom_triton_window_kernel",
    "custom_triton_window_reduction",
    "fused_triton_scan",
    "shape_specialized_triton_reduction",
    "single_pass_triton_scan",
    "specialized_triton_gemm",
    "specialized_triton_kernel",
    "specialized_triton_reduction",
    "tiled_triton_pair_reduction",
    "triangular_aware_triton",
    "triangular_sparse_triton",
    "triton-specialized-elementwise",
    "triton_custom_elementwise_kernel",
    "triton_fused_elementwise",
    "triton_fused_elementwise_kernel",
    "triton_fused_mse_reduction",
    "triton_fused_reduction",
    "triton_fused_triplet",
    "triton_row_reduction",
    "triton_scalar_elementwise",
    "triton_single_pass_reduction",
    "triton_specialized_reduction",
})

_BUILD_LANGUAGE_BY_STRATEGY_FAMILY = {
    strategy_family: "triton" for strategy_family in TRITON_STRATEGY_FAMILIES
}
_SUPPORTED_BUILD_LANGUAGES = frozenset({"python", "triton", "cuda", "cpp", "tilelang", "cute"})

_BUILD_HEADER = re.compile(r"(?m)^\[build\]\s*$")
_TABLE_HEADER = re.compile(r"(?m)^\[[^\]]+\]\s*$")
_LANGUAGE_LINE = re.compile(r"(?m)^language\s*=\s*[^\r\n]*(?P<newline>\r?\n|$)")


class BuildLanguageConfigurationError(RuntimeError):
    """A planned DSL strategy and the child build configuration disagree."""

    def __init__(self, strategy_family: str, expected: str, actual: str | None) -> None:
        self.strategy_family = strategy_family
        self.expected = expected
        self.actual = actual
        super().__init__(
            "Harness build-language configuration error: "
            f"strategy_family={strategy_family!r} requires build.language={expected!r}, "
            f"but child config has build.language={actual!r}."
        )


def resolve_build_language(strategy_family: str, declared_language: str | None = None) -> str | None:
    """Return the Harness-required build language, or preserve the child default."""
    if declared_language is not None:
        if declared_language not in _SUPPORTED_BUILD_LANGUAGES:
            raise ValueError(f"Unsupported planner build_language {declared_language!r}")
        return declared_language
    return _BUILD_LANGUAGE_BY_STRATEGY_FAMILY.get(strategy_family)


def read_build_language(config_path: Path) -> str | None:
    try:
        data = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    build = data.get("build")
    value = build.get("language") if isinstance(build, dict) else None
    return value if isinstance(value, str) else None


def materialize_build_language(config_path: Path, strategy_family: str, declared_language: str | None = None) -> str | None:
    """Write the planner-selected build language without delegating config to a worker."""
    expected = resolve_build_language(strategy_family, declared_language)
    if expected is None:
        return None
    path = Path(config_path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BuildLanguageConfigurationError(strategy_family, expected, None) from exc
    header = _BUILD_HEADER.search(source)
    if header is None:
        raise BuildLanguageConfigurationError(strategy_family, expected, None)
    next_header = _TABLE_HEADER.search(source, header.end())
    table_end = next_header.start() if next_header else len(source)
    table = source[header.end():table_end]
    replacement = f'language = "{expected}"\n'
    if _LANGUAGE_LINE.search(table):
        table = _LANGUAGE_LINE.sub(replacement, table, count=1)
    else:
        table = "\n" + replacement + table
    path.write_text(source[:header.end()] + table + source[table_end:], encoding="utf-8")
    return expected


def validate_build_language(config_path: Path, strategy_family: str, declared_language: str | None = None) -> None:
    """Fail before worker/benchmark execution when a required DSL backend is absent."""
    expected = resolve_build_language(strategy_family, declared_language)
    if expected is None:
        return
    actual = read_build_language(config_path)
    if actual != expected:
        raise BuildLanguageConfigurationError(strategy_family, expected, actual)
