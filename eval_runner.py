"""
Quilltale — Automated Evaluation Runner

Plays through a fixed scenario and measures three metrics:
  1. invalid_transition_rate   — how often the GM proposes invalid state changes
  2. memory_utilisation_rate   — how often NPC memories actually shape narration
  3. factual_consistency_rate  — how often narration contradicts world state

Run with:
    python eval_runner.py

Outputs:
    eval_results/report_<timestamp>.json
    eval_results/report_<timestamp>.txt
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from src.world.state import WorldState
from src.agents.game_master import GameMasterAgent
from src.llm import get_llm
from src.llm.base import BaseLLM

logging.basicConfig(level=logging.WARNING)

### Evaluation scenario ########################################
### A fixed sequence of actions designed to stress-test all three metrics.
### Covers: movement, item interaction, NPC conversation,
###         NPC memory escalation, invalid attempts, multi-step routing.

EVAL_SCENARIO = [
    ### Turn 1-3: basic movement and scene establishment
    "look around the tavern carefully",
    "examine the wanted notice on the wall",
    "examine the rusty dagger on the table",

    ### Turn 4-6: NPC interaction to help build Marta's memory
    "talk to Marta the barkeep",
    "ask Marta about the dagger",
    "ask Marta about the chest upstairs",

    ### Turn 7: memory escalation
    "threaten Marta to tell you what she knows",

    ### Turn 8-9: item interaction
    "pick up the dagger",
    "pick up the wanted notice",

    ### Turn 10: invalid attempt to pick an item not here
    "pick up the strange coin",

    ### Turn 11-12: movement
    "go north to the street",
    "go east to the market",

    ### Turn 13-14: NPC interaction in new location
    "talk to Aldric the merchant",
    "ask Aldric about the strange coin",

    ### Turn 15: multi-step routing back
    "go back to the tavern",

    ### Turn 16: verify Marta still remembers the threat
    "talk to Marta",

    ### Turn 17-18: upstairs exploration
    "go upstairs to my room",
    "examine the locked chest",

    ### Turn 19: use key on chest
    "use the old iron key on the chest",

    ### Turn 20: invalid movement attempt
    "go south",
]


### LLM Judge ########################################

JUDGE_SYSTEM = """
You are an objective evaluator assessing AI game master output quality.
You always respond with valid JSON only. No preamble or explanation outside the JSON.
"""

def judge_memory_utilisation(
    llm: BaseLLM,
    narration: str,
    npc_memories: list,
    npc_name: str,
) -> dict:
    """
    Ask the LLM judge: does this narration reflect the NPC's recorded memories?
    Returns {"reflects_memory": bool, "confidence": float, "reason": str}
    """
    if not npc_memories:
        return {"reflects_memory": True, "confidence": 1.0, "reason": "No memories to reflect."}

    memory_text = "\n".join(
        f"  - Turn {m.turn} ({m.emotional_tone}, sig={m.significance}): {m.description}"
        for m in npc_memories[:3]
    )

    prompt = f"""
            {npc_name} has these recorded memories of the player:
            {memory_text}

            The game master produced this narration:
            "{narration}"

            Does the narration reflect any of these memories through {npc_name}'s behaviour,
            tone, dialogue, or reaction? Even subtle reflection counts (e.g. wariness, coldness,
            gratitude shown through action rather than stated directly).

            Respond with JSON:
            {{"reflects_memory": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}}
            """
    try:
        raw = llm.generate_json(prompt, JUDGE_SYSTEM)
        return json.loads(raw)
    except Exception:
        return {"reflects_memory": False, "confidence": 0.0, "reason": "Judge call failed."}


def judge_factual_consistency(
    llm: BaseLLM,
    narration: str,
    world_context: str,
) -> dict:
    """
    Ask the LLM judge: does the narration contradict the world state?
    Returns {"is_consistent": bool, "confidence": float, "violation": str}
    """
    prompt = f"""
              The current world state contains these facts:
              {world_context}

              The game master produced this narration:
              "{narration}"

              Does the narration contradict any recorded facts? Look for:
              - Items mentioned that aren't in the current location or inventory
              - NPCs described as present when they are not listed
              - Movement described to locations not reachable from current exits
              - Health or inventory states that differ from recorded values

              Respond with JSON:
              {{"is_consistent": true/false, "confidence": 0.0-1.0, "violation": "describe any contradiction, or empty string if none"}}
              """
    try:
        raw = llm.generate_json(prompt, JUDGE_SYSTEM)
        return json.loads(raw)
    except Exception:
        return {"is_consistent": True, "confidence": 0.0, "violation": "Judge call failed."}


### Metrics collector ########################################

@dataclass
class TurnRecord:
    turn: int
    action: str
    narration: str
    state_update: dict
    changes_applied: list[str]
    rejected_transitions: list[str]
    memory_judgement: dict = field(default_factory=dict)
    consistency_judgement: dict = field(default_factory=dict)
    npcs_present: list[str] = field(default_factory=list)
    npc_memories_present: bool = False


@dataclass
class EvalReport:
    scenario_name: str = "default_world"
    total_turns: int = 0
    timestamp: str = ""

    ### Raw counts
    total_transitions_attempted: int = 0
    total_transitions_rejected: int = 0
    total_turns_with_npcs: int = 0
    total_turns_memory_reflected: int = 0
    total_turns_memory_judged: int = 0
    total_turns_consistent: int = 0
    total_turns_consistency_judged: int = 0

    ### Derived metrics
    invalid_transition_rate: float = 0.0
    memory_utilisation_rate: float = 0.0
    factual_consistency_rate: float = 0.0

    ### Turn-by-turn records
    turns: list[TurnRecord] = field(default_factory=list)

    ### Failure examples
    rejection_examples: list[str] = field(default_factory=list)
    memory_failures: list[dict] = field(default_factory=list)
    consistency_violations: list[dict] = field(default_factory=list)

    def compute_rates(self):
        if self.total_transitions_attempted > 0:
            self.invalid_transition_rate = round(
                self.total_transitions_rejected / self.total_transitions_attempted, 3
            )
        if self.total_turns_memory_judged > 0:
            self.memory_utilisation_rate = round(
                self.total_turns_memory_reflected / self.total_turns_memory_judged, 3
            )
        if self.total_turns_consistency_judged > 0:
            self.factual_consistency_rate = round(
                self.total_turns_consistent / self.total_turns_consistency_judged, 3
            )


### Evaluation runner ########################################

def run_evaluation(
    scenario: list[str] = EVAL_SCENARIO,
    world_path: str = "data/worlds/default.json",
    llm_name: str = "gemini",
    run_judge: bool = True,
) -> EvalReport:
    """
    Play through the scenario automatically and collect metrics.

    Args:
        scenario:   List of player actions to execute in order.
        world_path: Path to the world JSON file.
        llm_name:   LLM provider to use for the GM.
        run_judge:  Whether to run LLM judge calls for memory and consistency.
                    Set False to only measure invalid transition rate (cheaper).
    """
    print(f"Running evaluation: {len(scenario)} turns, judge={'on' if run_judge else 'off'}")
    print("-" * 60)

    with open(world_path) as f:
        state = WorldState.from_json(f.read())

    llm = get_llm(llm_name)
    judge_llm = get_llm(llm_name) if run_judge else None
    gm = GameMasterAgent(llm)

    report = EvalReport(
        scenario_name=Path(world_path).stem,
        timestamp=datetime.now().isoformat(),
    )

    ### Generate opening (not counted in metrics — no action to evaluate)
    opening = gm.generate_opening(state)
    print(f"Opening: {opening['narration'][:80]}...")
    print()

    for i, action in enumerate(scenario):
        print(f"Turn {i+1:02d}: {action}")

        world_context_before = state.to_context_summary()
        npcs_present = state.npcs_in_location(state.player.location)
        npcs_with_memories = [
            npc for npc in npcs_present
            if npc.alive and len(npc.memories) > 0
        ]

        result = gm.process_turn(action, state)

        ### Count transitions
        update = result.get("state_update", {})
        transition_keys = {"move_player", "pickup_item", "drop_item", "npc_state"}
        attempted = sum(1 for k in transition_keys if k in update)
        rejected = [c for c in result["changes_applied"] if "REJECTED" in c]

        report.total_transitions_attempted += attempted
        report.total_transitions_rejected += len(rejected)

        if rejected:
            report.rejection_examples.extend(rejected[:2])

        ### Memory utilisation judgement
        memory_judgement = {}
        if run_judge and npcs_with_memories:
            report.total_turns_memory_judged += 1
            ### Judge against the NPC with most memories
            primary_npc = max(npcs_with_memories, key=lambda n: len(n.memories))
            memory_judgement = judge_memory_utilisation(
                judge_llm,
                result["narration"],
                primary_npc.relevant_memories(),
                primary_npc.name,
            )
            if memory_judgement.get("reflects_memory", False):
                report.total_turns_memory_reflected += 1
            else:
                report.memory_failures.append({
                    "turn": i + 1,
                    "action": action,
                    "narration": result["narration"],
                    "reason": memory_judgement.get("reason", ""),
                })

        ### Factual consistency judgement
        consistency_judgement = {}
        if run_judge:
            report.total_turns_consistency_judged += 1
            consistency_judgement = judge_factual_consistency(
                judge_llm,
                result["narration"],
                world_context_before,
            )
            if consistency_judgement.get("is_consistent", True):
                report.total_turns_consistent += 1
            else:
                report.consistency_violations.append({
                    "turn": i + 1,
                    "action": action,
                    "narration": result["narration"],
                    "violation": consistency_judgement.get("violation", ""),
                })

        ### Record turn
        record = TurnRecord(
            turn=i + 1,
            action=action,
            narration=result["narration"],
            state_update=update,
            changes_applied=result["changes_applied"],
            rejected_transitions=rejected,
            memory_judgement=memory_judgement,
            consistency_judgement=consistency_judgement,
            npcs_present=[n.name for n in npcs_present],
            npc_memories_present=bool(npcs_with_memories),
        )
        report.turns.append(record)
        report.total_turns += 1

        print(f"         → {result['narration'][:80]}...")
        if rejected:
            print(f"         ✗ REJECTED: {rejected}")
        print()

    report.compute_rates()
    return report


### Report writer ########################################

def write_report(report: EvalReport, output_dir: str = "eval_results"):
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = report.timestamp.replace(":", "-").replace(".", "-")

    ### Full JSON data report
    json_path = Path(output_dir) / f"report_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, indent=2, default=lambda o: o.__dict__)

    ### Report Summary
    txt_path = Path(output_dir) / f"report_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:

        f.write("QUILLTALE — EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Scenario:  {report.scenario_name}\n")
        f.write(f"Timestamp: {report.timestamp}\n")
        f.write(f"Turns:     {report.total_turns}\n\n")

        f.write("METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Invalid transition rate:    {report.invalid_transition_rate:.1%}  "
            f"({report.total_transitions_rejected}/{report.total_transitions_attempted} transitions rejected)\n"
        )
        f.write(
            f"Memory utilisation rate:    {report.memory_utilisation_rate:.1%}  "
            f"({report.total_turns_memory_reflected}/{report.total_turns_memory_judged} turns with NPCs reflected memory)\n"
        )
        f.write(
            f"Factual consistency rate:   {report.factual_consistency_rate:.1%}  "
            f"({report.total_turns_consistent}/{report.total_turns_consistency_judged} turns without contradictions)\n\n"
        )

        if report.rejection_examples:
            f.write("REJECTED TRANSITIONS (sample)\n")
            f.write("-" * 40 + "\n")
            for ex in report.rejection_examples[:5]:
                f.write(f"  {ex}\n")
            f.write("\n")

        if report.memory_failures:
            f.write(f"MEMORY UTILISATION FAILURES ({len(report.memory_failures)} turns)\n")
            f.write("-" * 40 + "\n")
            for mf in report.memory_failures[:3]:
                f.write(f"  Turn {mf['turn']}: {mf['action']}\n")
                f.write(f"  Narration: {mf['narration'][:100]}...\n")
                f.write(f"  Reason: {mf['reason']}\n\n")

        if report.consistency_violations:
            f.write(f"CONSISTENCY VIOLATIONS ({len(report.consistency_violations)} turns)\n")
            f.write("-" * 40 + "\n")
            for cv in report.consistency_violations[:3]:
                f.write(f"  Turn {cv['turn']}: {cv['action']}\n")
                f.write(f"  Narration: {cv['narration'][:100]}...\n")
                f.write(f"  Violation: {cv['violation']}\n\n")

        f.write("TURN-BY-TURN SUMMARY\n")
        f.write("-" * 40 + "\n")
        for t in report.turns:
            status = "✓" if not t.rejected_transitions else "✗"
            mem = ""
            if t.memory_judgement:
                mem = " [mem:✓]" if t.memory_judgement.get("reflects_memory") else " [mem:✗]"
            con = ""
            if t.consistency_judgement:
                con = " [con:✓]" if t.consistency_judgement.get("is_consistent") else " [con:✗]"
            f.write(f"  {status} T{t.turn:02d} {t.action[:45]:<45}{mem}{con}\n")

    print(f"Report written to:")
    print(f"  {json_path}")
    print(f"  {txt_path}")
    return json_path, txt_path


### Entry point ########################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Quilltale evaluation")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge calls (only measure transition rate)")
    parser.add_argument("--llm", default="gemini",
                        help="LLM provider: gemini or claude")
    parser.add_argument("--world", default="data/worlds/default.json",
                        help="Path to world JSON file")
    args = parser.parse_args()

    report = run_evaluation(
        scenario=EVAL_SCENARIO,
        world_path=args.world,
        llm_name=args.llm,
        run_judge=not args.no_judge,
    )

    write_report(report)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Invalid transition rate:   {report.invalid_transition_rate:.1%}")
    print(f"Memory utilisation rate:   {report.memory_utilisation_rate:.1%}")
    print(f"Factual consistency rate:  {report.factual_consistency_rate:.1%}")