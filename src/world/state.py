"""
WorldState is the ground truth of the game world.
The LLM can never contradict what is recorded here.
All mutations go through update() ensuring no direct assignment.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import copy


@dataclass
class Item:
    id: str
    name: str
    description: str
    location: str           # location_id or "inventory" or NPC id
    properties: dict = field(default_factory=dict)


@dataclass
class MemoryEntry:
    """
    A single episodic memory belonging to an NPC.

    significance: 1-3
        1 = minor (player asked about the weather)
        2 = notable (player helped or insulted them)
        3 = significant (player attacked, betrayed, or saved them)
    """
    turn: int
    description: str          # "Player picked up the dagger without asking"
    emotional_tone: str       # "suspicious" | "grateful" | "fearful" | "angry" | "neutral"
    significance: int = 1     # 1-3


@dataclass
class NPC:
    id: str
    name: str
    description: str
    location: str           # location_id
    disposition: str        # "friendly" | "neutral" | "hostile"
    alive: bool = True
    inventory: list[str] = field(default_factory=list)  # item ids
    memories: list[MemoryEntry] = field(default_factory=list)

    def relevant_memories(self, max_memories: int = 5) -> list[MemoryEntry]:
        """
        Return the most relevant memories for prompt injection.
        Priority: significant memories first, then most recent.
        Capped to avoid bloating the prompt.
        """
        sorted_memories = sorted(
            self.memories,
            key=lambda m: (m.significance, m.turn),
            reverse=True
        )
        return sorted_memories[:max_memories]

    def memory_summary(self) -> str:
        """
        Compact string for prompt injection.
        Only called when this NPC is in the current location.
        """
        relevant = self.relevant_memories()
        if not relevant:
            return f"{self.name} has no prior interactions with the player."

        lines = [f"{self.name}'s memory of the player:"]
        for m in relevant:
            lines.append(f"  - Turn {m.turn} ({m.emotional_tone}): {m.description}")
        return "\n".join(lines)

@dataclass
class Location:
    id: str
    name: str
    description: str
    exits: dict[str, str]   # {"north": "location_id", "east": "location_id"}
    items: list[str] = field(default_factory=list)   # item ids
    npcs: list[str] = field(default_factory=list)    # npc ids
    visited: bool = False


@dataclass
class Player:
    location: str           # location_id
    inventory: list[str] = field(default_factory=list)  # item ids
    health: int = 100
    known_facts: list[str] = field(default_factory=list)


@dataclass
class WorldState:
    """
    Single source of truth for the entire game world.
    """
    locations: dict[str, Location] = field(default_factory=dict)
    npcs: dict[str, NPC] = field(default_factory=dict)
    items: dict[str, Item] = field(default_factory=dict)
    player: Player = field(default_factory=lambda: Player(location="start"))
    turn: int = 0
    world_name: str = "Unknown World"
    history: list[dict] = field(default_factory=list)  # {turn, action, narration}

    def current_location(self) -> Optional[Location]:
        return self.locations.get(self.player.location)

    def items_in_location(self, location_id: str) -> list[Item]:
        loc = self.locations.get(location_id)
        if not loc:
            return []
        return [self.items[i] for i in loc.items if i in self.items]

    def npcs_in_location(self, location_id: str) -> list[NPC]:
        loc = self.locations.get(location_id)
        if not loc:
            return []
        return [self.npcs[n] for n in loc.npcs if n in self.npcs]

    def player_inventory_items(self) -> list[Item]:
        return [self.items[i] for i in self.player.inventory if i in self.items]

    def to_context_summary(self) -> str:
        """
        Produces a structured context block for the LLM prompt.
        Includes NPC episodic memories when NPCs are present in the scene.
        """
        loc = self.current_location()
        if not loc:
            return "ERROR: Player location not found."

        loc_items = self.items_in_location(self.player.location)
        loc_npcs = self.npcs_in_location(self.player.location)
        player_items = self.player_inventory_items()

        exits_str = ", ".join(
            f"{d} → {self.locations[lid].name}"
            for d, lid in loc.exits.items()
            if lid in self.locations
        ) or "none"

        items_str = ", ".join(f"{i.name} [id:{i.id}]" for i in loc_items) or "none"

        # NPC summary with disposition AND memory
        npc_lines = []
        for n in loc_npcs:
            status = "alive" if n.alive else "dead"
            npc_lines.append(f"{n.name} [id:{n.id}] ({n.disposition}, {status})")
        npcs_str = ", ".join(npc_lines) or "none"

        inv_str = ", ".join(f"{i.name} [id:{i.id}]" for i in player_items) or "nothing"

        # Build NPC memory block, only if NPCs are present
        memory_block = ""
        if loc_npcs:
            memory_lines = []
            for npc in loc_npcs:
                if npc.alive:
                    memory_lines.append(f"[id:{npc.id}] {npc.memory_summary()}")
            if memory_lines:
                memory_block = "\nNPC MEMORIES (what they remember about the player):\n"
                memory_block += "\n".join(memory_lines)

        return f"""WORLD STATE (TURN {self.turn}) — THESE FACTS CANNOT BE CONTRADICTED:
                Location: {loc.name}
                Description: {loc.description}
                Exits: {exits_str}
                Items here: {items_str}
                NPCs here: {npcs_str}
                Player health: {self.player.health}/100
                Player inventory: {inv_str}{memory_block}
                """


    def to_player_summary(self) -> str:
        """
        Snapshot for the UI.
        """
        loc = self.current_location()
        if not loc:
            return "Location unknown."

        loc_items = self.items_in_location(self.player.location)
        loc_npcs = self.npcs_in_location(self.player.location)
        player_items = self.player_inventory_items()

        exits = ", ".join(
            f"{d} to {self.locations[lid].name}"
            for d, lid in loc.exits.items()
            if lid in self.locations
        ) or "none"

        items = ", ".join(i.name for i in loc_items) or "nothing"
        npcs = ", ".join(
            f"{n.name} ({n.disposition})"
            for n in loc_npcs if n.alive
        ) or "nobody"

        visited = [
            loc.name for loc_id, loc in self.locations.items()
            if loc.visited and loc_id != self.player.location
        ]

        visited_str = ", ".join(visited) if visited else "nowhere yet"

        carrying = ", ".join(i.name for i in player_items) or "nothing"

        facts = ""
        if self.player.known_facts:
            facts = "\n\nThings you know:\n" + "\n".join(
                f"  {f}" for f in self.player.known_facts
            )

        return (
            f"Turn {self.turn}\n\n"
            f"You are in {loc.name}.\n"
            f"{loc.description}\n\n"
            f"You can go: {exits}\n"
            f"You can see: {items}\n"
            f"Present: {npcs}\n"
            f"You have visited: {visited_str}\n"
            f"You are carrying: {carrying}"
            f"{facts}"
        )


    def to_map_summary(self) -> str:
        """
        Map of visited locations only, for GM route planning.
        Unvisited locations stay hidden until the player discovers them.
        """
        lines = ["KNOWN MAP (locations the player has visited):"]
        for loc_id, loc in self.locations.items():
            if not loc.visited:
                continue
            exits = ", ".join(
                f"{direction} → {self.locations[lid].name}"
                for direction, lid in loc.exits.items()
                if lid in self.locations
            ) or "no exits"
            lines.append(f"  {loc.name}: {exits}")

        if len(lines) == 1:
            return "KNOWN MAP: nowhere explored yet beyond current location."
        return "\n".join(lines)


    def apply_update(self, update: dict) -> list[str]:
        """
        Apply a structured update from the GM agent.
        Returns a list of change descriptions for logging.
        Validates before applying and rejects invalid transitions.
        """
        changes = []

        # Player movement
        if "move_player" in update:
            dest = update["move_player"]

            # Normalise to list to handle both single direction and multi-step
            directions = dest if isinstance(dest, list) else [dest]

            # For multi-step paths, check if final destination is visited
            # If not visited, truncate to single step only
            if len(directions) > 1:
                # Simulate the path to find the final location
                simulated_loc = self.player.location
                valid_directions = []
                for direction in directions:
                    loc = self.locations.get(simulated_loc)
                    if loc and direction in loc.exits and loc.exits[direction] in self.locations:
                        next_loc_id = loc.exits[direction]
                        next_loc = self.locations[next_loc_id]
                        if not next_loc.visited and next_loc_id != self.player.location:
                            # Unvisited destination — stop path here, take only this one step
                            valid_directions.append(direction)
                            break
                        valid_directions.append(direction)
                        simulated_loc = next_loc_id
                    else:
                        break
                directions = valid_directions if valid_directions else directions[:1]

            for direction in directions:
                loc = self.current_location()
                if direction in loc.exits and loc.exits[direction] in self.locations:
                    new_loc_id = loc.exits[direction]
                    self.player.location = new_loc_id
                    self.locations[new_loc_id].visited = True
                    changes.append(f"Player moved to {self.locations[new_loc_id].name}")
                else:
                    changes.append(f"REJECTED move to {direction} — not a valid exit")
                    break

        # Item pickup
        if "pickup_item" in update:
            item_id = update["pickup_item"]
            loc = self.current_location()
            if item_id in loc.items and item_id in self.items:
                loc.items.remove(item_id)
                self.player.inventory.append(item_id)
                changes.append(f"Player picked up {self.items[item_id].name}")
            else:
                changes.append(f"REJECTED pickup {item_id} — not in current location")

        # Item drop
        if "drop_item" in update:
            item_id = update["drop_item"]
            if item_id in self.player.inventory:
                self.player.inventory.remove(item_id)
                self.current_location().items.append(item_id)
                changes.append(f"Player dropped {self.items[item_id].name}")

        # NPC state change
        if "npc_state" in update:
            for npc_id, new_state in update["npc_state"].items():
                if npc_id in self.npcs:
                    npc = self.npcs[npc_id]
                    if "alive" in new_state:
                        npc.alive = new_state["alive"]
                    if "disposition" in new_state:
                        npc.disposition = new_state["disposition"]
                    changes.append(f"NPC {npc.name} state updated: {new_state}")

        # Write NPC memory
        if "npc_memory" in update:
            for npc_id, memory_data in update["npc_memory"].items():
                if npc_id in self.npcs:
                    npc = self.npcs[npc_id]
                    entry = MemoryEntry(
                        turn=self.turn,
                        description=memory_data.get("description", ""),
                        emotional_tone=memory_data.get("emotional_tone", "neutral"),
                        significance=int(memory_data.get("significance", 1)),
                    )
                    npc.memories.append(entry)
                    changes.append(
                        f"{npc.name} remembers: '{entry.description}' "
                        f"[{entry.emotional_tone}, significance {entry.significance}]"
                    )

        # Player health
        if "player_health" in update:
            delta = update["player_health"]
            self.player.health = max(0, min(100, self.player.health + delta))
            changes.append(f"Player health: {self.player.health}/100")

        # New fact learned
        if "add_fact" in update:
            self.player.known_facts.append(update["add_fact"])
            changes.append(f"Fact learned: {update['add_fact']}")

        self.turn += 1
        return changes

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "WorldState":
        raw = json.loads(data)
        # Reconstruct nested dataclasses
        state = cls()
        state.world_name = raw.get("world_name", "Unknown")
        state.turn = raw.get("turn", 0)
        state.history = raw.get("history", [])
        state.player = Player(**raw["player"])
        state.locations = {}
        for k, v in raw["locations"].items():
            loc_data = dict(v)
            loc_data.setdefault("visited", False)
            loc_data.setdefault("items", [])
            loc_data.setdefault("npcs", [])
            state.locations[k] = Location(**loc_data)
        state.npcs = {}
        for k, v in raw["npcs"].items():
            npc_data = dict(v)
            # Deserialise MemoryEntry objects from raw dicts
            raw_memories = npc_data.pop("memories", [])
            npc = NPC(**npc_data)
            npc.memories = [MemoryEntry(**m) for m in raw_memories]
            state.npcs[k] = npc
        state.items = {k: Item(**v) for k, v in raw["items"].items()}
        return state
