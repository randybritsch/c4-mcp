import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control4_gateway import Control4Gateway


def main() -> None:
    gw = Control4Gateway()
    director = gw._loop_thread.run(gw._director_async(), timeout_s=20)

    names = [n for n in dir(director) if "item" in n.lower() and callable(getattr(director, n))]
    print(f"Director item-related methods: {len(names)}")
    for name in sorted(names):
        fn = getattr(director, name)
        try:
            sig = str(inspect.signature(fn))
        except Exception as e:
            sig = f"<no signature: {type(e).__name__}: {e}>"
        print(f"  {name}{sig}")


if __name__ == "__main__":
    main()
