# import_test.py
from control4_adapter import list_rooms

rooms = list_rooms()
print(f"Rooms ({len(rooms)}):")
for r in rooms:
    print(r)
