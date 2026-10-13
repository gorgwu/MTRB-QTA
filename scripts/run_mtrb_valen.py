"""Run MTRB-MetaTool through JiuwenSwarm with a local Valen System One endpoint.

Set MTRB_TOOL_DISCOVERY_MODEL=valen-4b and the Valen endpoint/key overrides,
then invoke this script. It shares the validated Jev-compatible retrieval
runner and writes to a Valen-specific report by default.
"""

import os

from run_mtrb_jev import main


if __name__ == "__main__":
    model = os.getenv("MTRB_TOOL_DISCOVERY_MODEL", "")
    endpoint = os.getenv("MTRB_TOOL_DISCOVERY_API_BASE", "")
    api_key = os.getenv("MTRB_TOOL_DISCOVERY_API_KEY", "")
    if "valen" not in model.lower():
        raise SystemExit("Set MTRB_TOOL_DISCOVERY_MODEL=valen-4b before running Valen.")
    if not endpoint:
        raise SystemExit("Set MTRB_TOOL_DISCOVERY_API_BASE to the Valen /v1/systemone endpoint.")
    if not api_key:
        raise SystemExit("Set MTRB_TOOL_DISCOVERY_API_KEY to the Valen server API key.")
    raise SystemExit(main())
