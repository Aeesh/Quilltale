# tests/test_world_state.py

from src.world.state import WorldState
import json

def load_test_state() -> WorldState:
    with open("data/worlds/default.json") as f:
        return WorldState.from_json(f.read())


def test_valid_movement():
    state = load_test_state()
    assert state.player.location == "tavern"
    changes = state.apply_update({"move_player": "north"})
    assert state.player.location == "street"
    assert any("moved" in c for c in changes)


def test_invalid_movement_rejected():
    state = load_test_state()
    changes = state.apply_update({"move_player": "south"})
    # "south" is not a valid exit from tavern
    assert state.player.location == "tavern"
    assert any("REJECTED" in c for c in changes)


def test_valid_item_pickup():
    state = load_test_state()
    changes = state.apply_update({"pickup_item": "dagger"})
    assert "dagger" in state.player.inventory
    assert "dagger" not in state.locations["tavern"].items


def test_invalid_pickup_rejected():
    state = load_test_state()
    # strange_coin is in market, not tavern
    changes = state.apply_update({"pickup_item": "strange_coin"})
    assert "strange_coin" not in state.player.inventory
    assert any("REJECTED" in c for c in changes)


def test_serialisation_roundtrip():
    state = load_test_state()
    state.apply_update({"move_player": "north"})
    state.apply_update({"pickup_item": "dagger"})
    serialised = state.to_json()
    restored = WorldState.from_json(serialised)
    assert restored.player.location == state.player.location
    assert restored.player.inventory == state.player.inventory
    assert restored.turn == state.turn


def test_context_summary_contains_facts():
    state = load_test_state()
    summary = state.to_context_summary()
    assert "Broken Flagon" in summary
    assert "north" in summary.lower()
    assert "dagger" in summary.lower()