import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control4_adapter import get_all_items


def main() -> None:
    items = get_all_items()
    print(f"items: {len(items)}")
    locks = [
        i
        for i in items
        if isinstance(i, dict) and i.get("typeName") == "device" and (i.get("control") or "") == "lock"
    ]
    print(f"locks: {len(locks)}")
    for lock in sorted(locks, key=lambda x: (str(x.get("name") or ""), int(x.get("id") or 0))):
        print(f"{lock.get('id')}\t{lock.get('name')}\troom={lock.get('parentId')}")


if __name__ == "__main__":
    main()
