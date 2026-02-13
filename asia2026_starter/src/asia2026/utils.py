from __future__ import annotations

import json
import os
import random
import string
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str) -> str:
    """Unique-ish run id with timestamp + Slurm job id (if present) + random suffix."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job = os.environ.get("SLURM_JOB_ID", "nojid")
    rand = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    return f"{prefix}__{ts}__job{job}__{rand}"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class RunConfig:
    track: int
    method: str
    data_root: str
    run_root: str
    seed: int
    do_cv: bool
    n_splits: int
    limit_rows: Optional[int] = None
    limit_targets: Optional[int] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
