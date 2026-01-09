import asyncio
import json
from pathlib import Path

from pyControl4.account import C4Account
from pyControl4.director import C4Director


def load_config() -> dict:
    cfg_path = Path(__file__).with_name("config.json")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


async def main():
    cfg = load_config()
    host = cfg["host"]
    username = cfg["username"]
    password = cfg["password"]

    # 1) Authenticate to Control4 account API
    account = C4Account(username, password)
    await account.getAccountBearerToken()

    # 2) Get registered controller info
    controller_info = await account.getAccountControllers()
    controller_name = controller_info["controllerCommonName"]
    print(f"Controller name: {controller_name}")

    # 3) Get Director token for local controller access
    director_token_resp = await account.getDirectorBearerToken(controller_name)
    director_token = director_token_resp["token"]

    # 4) Connect to Director and fetch all items (rooms/devices/etc.)
    director = C4Director(host, director_token)
    items = await director.getAllItemInfo()

    print("Type of items:", type(items))

    # If Director returns JSON as a string, parse it
    if isinstance(items, str):
        print("Length of JSON string:", len(items))
        print("First 200 chars:", items[:200])

        try:
            items_obj = json.loads(items)
        except Exception as e:
            print("json.loads failed:", repr(e))
            return
    else:
        items_obj = items

    print("\nParsed type:", type(items_obj))
    if isinstance(items_obj, dict):
        print("Top-level keys:", list(items_obj.keys())[:50])

        # Try common containers where the real list lives
        for k in ["items", "itemInfo", "item_info", "data", "result"]:
            if k in items_obj:
                v = items_obj[k]
                print(f"\nFound '{k}' type:", type(v))
                if isinstance(v, list):
                    print(f"'{k}' list length:", len(v))
                    print("First element type:", type(v[0]) if v else None)
                    if v and isinstance(v[0], dict):
                        print("First element keys:", list(v[0].keys())[:50])
                elif isinstance(v, dict):
                    print(f"'{k}' dict keys sample:", list(v.keys())[:50])

    elif isinstance(items_obj, list):
        print("List length:", len(items_obj))
        print("First element type:", type(items_obj[0]) if items_obj else None)
        if items_obj and isinstance(items_obj[0], dict):
            print("First element keys:", list(items_obj[0].keys())[:50])
    # Extract rooms
    rooms = [i for i in items_obj if i.get("typeName") == "room"]
    print(f"\nRooms found: {len(rooms)}")
    for r in rooms:
        print(f"- room id={r.get('id')} name={r.get('name')} parentId={r.get('parentId')}")


if __name__ == "__main__":
    asyncio.run(main())
