import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control4_gateway import Control4Gateway


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python tools/inspect_item.py <item_id>")
        raise SystemExit(2)

    item_id = int(sys.argv[1])
    gw = Control4Gateway()
    director = gw._loop_thread.run(gw._director_async(), timeout_s=20)

    info = None
    if hasattr(director, "getItemInfo"):
        try:
            info = gw._loop_thread.run(director.getItemInfo(item_id), timeout_s=10)
        except Exception as e:
            info = {"_error": str(e), "_error_type": type(e).__name__}

    cmds = None
    if hasattr(director, "getItemCommands"):
        try:
            cmds = gw._loop_thread.run(director.getItemCommands(item_id), timeout_s=10)
        except Exception as e:
            cmds = {"_error": str(e), "_error_type": type(e).__name__}

    vars_ = None
    if hasattr(director, "getItemVariables"):
        try:
            vars_ = gw._loop_thread.run(director.getItemVariables(item_id), timeout_s=10)
        except Exception as e:
            vars_ = {"_error": str(e), "_error_type": type(e).__name__}

    def _maybe_json(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return x
        return x

    info = _maybe_json(info)
    cmds = _maybe_json(cmds)
    vars_ = _maybe_json(vars_)

    out = {
        "item_id": item_id,
        "info": info,
        "commands": cmds,
        "variables": vars_,
    }
    print(json.dumps(out, indent=2)[:12000])


if __name__ == "__main__":
    main()
