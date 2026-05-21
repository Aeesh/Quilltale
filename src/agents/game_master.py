"""
The Game Master agent.

Every turn:
1. Reads the current WorldState as a structured context block
2. Generates narrative text that CANNOT contradict recorded facts
3. Returns a structured JSON update specifying what changed
4. The WorldState applies the update and rejects invalid transitions

The LLM is responsible for story quality.
The WorldState is responsible for factual consistency.
These two concerns are cleanly separated.
"""

import json
import logging
from src.llm.base import BaseLLM
from src.world.state import WorldState

logger = logging.getLogger(__name__)

GM_SYSTEM = """
You are a game master running a dark, grounded text adventure.

YOUR RULES:
1. You receive the WORLD STATE before every response. These facts are absolute.
   You cannot invent items, exits, or NPCs that are not listed there.
   You cannot move the player without a valid exit.
   You cannot resurrect dead NPCs or change their known disposition without a state update.

2. NPC MEMORY RULES — this is what makes your NPCs intelligent:
   - When NPCs are present and a significant interaction occurs, you MUST write
     a memory entry for that NPC.
   - Significant = anything the NPC would actually remember: being attacked,
     helped, lied to, given something, threatened, ignored when spoken to,
     overheard saying something, etc.
   - Minor = routine pleasantries, passing through, looking around.
   - NPCs use their memories to inform how they speak and behave.
     If the barkeep remembers the player stole from her, she is cold and watchful.
     If the merchant remembers being paid fairly, he offers a small discount.
   - Memory shapes tone, not just dialogue, so show it through behaviour.
   - You MUST include npc_memory in state_update whenever an NPC is present and
     any interaction occurred, even minor ones. Never omit it when NPCs are in the scene.

3. Your response has two parts: what the player sees (narration, image_prompt) and what the world records
(state_update, scene_changed), always in this exact JSON structure:
{
  "narration": "The story text the player reads. 2-4 sentences. Grounded in facts.",
  "state_update": {
    "move_player": "direction (only if player explicitly tried to move)",
    "pickup_item": "item_id (only if player explicitly picked it up)",
    "drop_item": "item_id (only if player explicitly dropped it)",
    "npc_state": {"npc_id": {"alive": bool, "disposition": "friendly|neutral|hostile"}},
    "npc_memory": {
      "npc_id": {
        "description": "One sentence: what the NPC now remembers about this interaction",
        "emotional_tone": "suspicious|grateful|fearful|angry|amused|neutral|wary",
        "significance": 1
      }
    },
    "player_health": -10,
    "add_fact": "a short string describing something the player learned"
  },
  "scene_changed": true,
  "image_prompt": "A cinematic description of the current scene for image generation. Only when scene_changed is true."
}

4. significance scale for npc_memory:
   1 = minor (asked a question, walked past, bought something routine)
   2 = notable (helped them, insulted them, took something, made a promise)
   3 = significant (attacked them, saved their life, betrayed their trust, revealed a secret)

5. Only write npc_memory when an NPC is present AND an actual interaction occurred.
   Do not write memory for NPCs in other locations.
   Do not write memory for dead NPCs.

6. state_update fields are optional — only include what actually changed.

7. image_prompt should only appear when scene_changed is true.
   It should describe the visual scene in rich, concrete terms: lighting, mood,
   architecture, characters present, time of day. Do not include player stats.

8. Write narration in second person. Keep it atmospheric, specific, and short.
   Let NPC behaviour reflect their memories, do not explain why they act that way,
   just show it.
"""

GM_PROMPT = """
{world_context}

RECENT HISTORY:
{history}

PLAYER ACTION: {action}

Respond with the JSON structure described in your instructions.
"""


class GameMasterAgent:
    def __init__(self, llm: BaseLLM):
        self._llm = llm

    def process_turn(self, action: str, state: WorldState) -> dict:
        """
        Process one player action. Returns dict with:
        - narration: str
        - state_update: dict
        - scene_changed: bool
        - image_prompt: str | None
        - changes_applied: list[str]
        """
        history_str = self._format_history(state.history[-5:])

        prompt = GM_PROMPT.format(
            world_context=state.to_context_summary(),
            history=history_str,
            action=action,
        )

        try:
            raw = self._llm.generate_json(prompt, GM_SYSTEM)
            result = json.loads(raw)
        except Exception as e:
            logger.error(f"GM parse error: {e}\nRaw: {raw if 'raw' in dir() else 'N/A'}")
            return {
                "narration": "Something shifts in the air, but you cannot tell what.",
                "state_update": {},
                "scene_changed": False,
                "image_prompt": None,
                "changes_applied": [],
            }

        # Apply state update — WorldState validates and rejects bad transitions
        changes = state.apply_update(result.get("state_update", {}))

        # Log to history
        state.history.append({
            "turn": state.turn,
            "action": action,
            "narration": result.get("narration", ""),
        })

        return {
            "narration": result.get("narration", ""),
            "state_update": result.get("state_update", {}),
            "scene_changed": result.get("scene_changed", False),
            "image_prompt": result.get("image_prompt"),
            "changes_applied": changes,
        }

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "None yet."
        return "\n".join(
            f"Turn {h['turn']}: [{h['action']}] → {h['narration']}"
            for h in history
        )

    def generate_opening(self, state: WorldState) -> dict:
        """Generate the opening narration when a new game starts."""
        loc = state.current_location()
        prompt = f"""
              {state.to_context_summary()}

              Generate the opening narration for this adventure. Set the scene.
              Respond with JSON: {{"narration": "...", "image_prompt": "..."}}
              """
        try:
            raw = self._llm.generate_json(prompt, GM_SYSTEM)
            result = json.loads(raw)
            return {
                "narration": result.get("narration", f"You find yourself in {loc.name}."),
                "image_prompt": result.get("image_prompt"),
                "scene_changed": True,
            }
        except Exception:
            return {
                "narration": f"You find yourself in {loc.name if loc else 'an unknown place'}.",
                "image_prompt": None,
                "scene_changed": True,
            }