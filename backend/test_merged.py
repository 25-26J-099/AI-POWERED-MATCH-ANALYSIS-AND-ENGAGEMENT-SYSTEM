"""Quick test for merged.py logic (no Ollama or TTS needed)."""
import sys
sys.path.insert(0, r"d:\3BMM2")

print("1. Importing modules...")
import tac_commentary as tac
import pbp_commentary as pbp
import json

print("2. Loading data...")
events = json.load(open(r"d:\3BMM2\demo1_event.json", "r", encoding="utf-8"))
threesixty = json.load(open(r"d:\3BMM2\demo1_threesixty.json", "r", encoding="utf-8"))
lookup = {f.get("event_uuid"): f for f in threesixty if f.get("event_uuid")}
print(f"   Events: {len(events)}, 360 frames: {len(lookup)}")

print("3. Base timestamp...")
base_ts = tac.get_event_clock_seconds(events[0])
print(f"   Base TS: {base_ts}")

print("4. Tactical plan...")
plan = tac.build_commentary_plan(events, lookup)
print(f"   TAC plan: {len(plan)} events")
for item in plan:
    eid = item["event_id"][:12]
    reason = item["selection_reason"]
    print(f"     - {eid}... ({reason})")

print("5. PBP anchors...")
anchors = pbp.detect_all_anchors(events, lookup)
print(f"   PBP anchors: {len(anchors)} events")
for a in anchors[:5]:
    etype = pbp.get_event_type(a["event"])
    player = pbp.get_player_last_name(a["event"])
    print(f"     - {etype} by {player} ({a['reason']})")

print("\n=== ALL TESTS PASSED ===")
