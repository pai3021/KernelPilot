"""Immutable-on-write candidate witness captured immediately before evaluation."""
from __future__ import annotations
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# 来源记录
def capture_evaluated_candidate(workspace: Path) -> dict[str, str]:
    """Copy the exact kernel about to be packed and bind it to a SHA-256."""
    # 读取候选代码，计算 SHA-256
    source = workspace / "solution" / "kernel.py"
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    captured_at = datetime.now(timezone.utc).isoformat()
    # 用哈希命名，保存一份代码快照
    directory = workspace / ".ako" / "evaluated-candidates"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / f"{digest}.kernel.py"
    # 同内容复用同名快照；这里不覆盖已有快照，也不设置文件只读权限。
    if not snapshot.exists():
        shutil.copy2(source, snapshot)
    # 写入最新捕获版本的指针；旧快照保留，时间为本地捕获时的 UTC 时间。
    record = {"candidate_sha256": digest, "artifact_path": str(snapshot.resolve()), "evaluation_timestamp": captured_at}
    (workspace / ".ako" / "evaluated-candidate.json").write_text(json.dumps(record, indent=2) + "\n")
    # runner 随后另行打包工作区；这里没有锁住源码，也没有直接从快照生成评测包。
    return record
