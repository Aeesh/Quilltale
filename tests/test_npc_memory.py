"""
Tests for NPC episodic memory system.
"""

from src.world.state import WorldState, MemoryEntry
import json


def load_test_state() -> WorldState:
    with open("data/worlds/default.json") as f:
        return WorldState.from_json(f.read())


def test_memory_written_on_interaction():
    state = load_test_state()
    changes = state.apply_update({
        "npc_memory": {
            "barkeep": {
                "description": "Player asked about the locked chest upstairs",
                "emotional_tone": "suspicious",
                "significance": 2,
            }
        }
    })
    barkeep = state.npcs["barkeep"]
    assert len(barkeep.memories) == 1
    assert barkeep.memories[0].emotional_tone == "suspicious"
    assert barkeep.memories[0].significance == 2
    assert any("Marta" in c for c in changes)


def test_multiple_memories_accumulate():
    state = load_test_state()

    state.apply_update({"npc_memory": {
        "barkeep": {"description": "Player ordered ale", "emotional_tone": "neutral", "significance": 1}
    }})
    state.apply_update({"npc_memory": {
        "barkeep": {"description": "Player stole the dagger", "emotional_tone": "angry", "significance": 3}
    }})
    state.apply_update({"npc_memory": {
        "barkeep": {"description": "Player apologised and returned the dagger", "emotional_tone": "wary", "significance": 2}
    }})

    barkeep = state.npcs["barkeep"]
    assert len(barkeep.memories) == 3


def test_relevant_memories_prioritises_significance():
    state = load_test_state()
    barkeep = state.npcs["barkeep"]

    # Add memories with different significance levels
    barkeep.memories = [
        MemoryEntry(turn=1, description="minor thing", emotional_tone="neutral", significance=1),
        MemoryEntry(turn=2, description="very significant thing", emotional_tone="angry", significance=3),
        MemoryEntry(turn=3, description="notable thing", emotional_tone="wary", significance=2),
    ]

    relevant = barkeep.relevant_memories(max_memories=2)
    # Should get the significance=3 memory first
    assert relevant[0].significance == 3
    assert relevant[1].significance == 2


def test_memory_appears_in_context_summary():
    state = load_test_state()

    state.apply_update({"npc_memory": {
        "barkeep": {
            "description": "Player threatened her with the dagger",
            "emotional_tone": "fearful",
            "significance": 3,
        }
    }})

    summary = state.to_context_summary()
    # Memory should appear in the context the GM sees
    assert "threatened" in summary
    assert "fearful" in summary


def test_memory_survives_serialisation_roundtrip():
    state = load_test_state()

    state.apply_update({"npc_memory": {
        "barkeep": {
            "description": "Player left a large tip",
            "emotional_tone": "grateful",
            "significance": 2,
        }
    }})

    # Serialise and restore
    restored = WorldState.from_json(state.to_json())

    barkeep = restored.npcs["barkeep"]
    assert len(barkeep.memories) == 1
    assert barkeep.memories[0].emotional_tone == "grateful"
    assert barkeep.memories[0].description == "Player left a large tip"


def test_dead_npc_memory_not_injected_in_summary():
    state = load_test_state()

    # Give barkeep a memory then kill her
    state.apply_update({"npc_memory": {
        "barkeep": {"description": "Player was rude", "emotional_tone": "angry", "significance": 1}
    }})
    state.apply_update({"npc_state": {"barkeep": {"alive": False}}})

    summary = state.to_context_summary()
    # Dead NPCs should not inject memories
    assert "Player was rude" not in summary


def test_memory_summary_format():
    state = load_test_state()
    barkeep = state.npcs["barkeep"]

    barkeep.memories = [
        MemoryEntry(turn=5, description="Gave player information about the chest",
                   emotional_tone="neutral", significance=1),
    ]

    summary = barkeep.memory_summary()
    assert "Marta" in summary
    assert "Gave player information" in summary
    assert "neutral" in summary