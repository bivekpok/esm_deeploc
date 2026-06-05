#!/usr/bin/env python3
"""Write a frozen experiment snapshot for Slurm jobs (see config.apply_run_snapshot)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import config, run_settings_dict, sync_run_paths  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <snapshot.json>")
    out = Path(sys.argv[1])
    sync_run_paths(cfg=config)
    payload = run_settings_dict(cfg=config)
    payload["snapshot_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["wandb_project"] = config.wandb_project
    payload["wandb_mode"] = config.wandb_mode
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote run snapshot: {out}")


if __name__ == "__main__":
    main()
