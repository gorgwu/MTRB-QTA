"""Run MTRB-MetaTool against a local Laya System One endpoint."""

from __future__ import annotations

import os

from run_mtrb_jev import main


if __name__ == "__main__":
    os.environ.setdefault("MTRB_TOOL_DISCOVERY_MODEL", "english")
    os.environ.setdefault("MTRB_TOOL_DISCOVERY_MAX_LEN", "8192")
    os.environ.setdefault("MTRB_TOOL_DISCOVERY_HEAD_MAX_LEN", "7800")

    model = os.environ["MTRB_TOOL_DISCOVERY_MODEL"].lower()
    if "laya" not in model and model != "english":
        raise SystemExit(
            "Set MTRB_TOOL_DISCOVERY_MODEL to a Laya checkpoint name (default: english)."
        )

    raise SystemExit(main())
