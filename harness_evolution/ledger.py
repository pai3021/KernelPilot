"""Small JSONL ledger and monotonic H0/H1 version marker."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def read_version(path: Path) -> str:
    # 当前逻辑版本保存在 JSON；尚未建立版本文件时，以 H0 为起点。
    if not path.is_file():
        return "H0"
    return str(json.loads(path.read_text())["version"])


def write_version(path: Path, version: str) -> None:
    # 覆盖保存当前版本标记；不备份模板内容，也不创建 Git 提交。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": version}, indent=2) + "\n")


def next_version(version: str) -> str:
    # 校验 H+数字 的形式并递增，例如 H0 -> H1；这里只返回字符串，不写文件。
    if not version.startswith("H") or not version[1:].isdigit():
        raise ValueError(f"invalid harness version: {version}")
    return f"H{int(version[1:]) + 1}"


def append_ledger(path: Path, record: Mapping[str, Any]) -> None:
    # JSONL 每行一条处理历史，包含 campaign、提案、决策及修改前后版本等。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 默认补入当前 UTC 时间；record 自带 timestamp 时会覆盖此默认值。
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
