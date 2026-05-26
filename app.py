"""
Quilltale — AI Game Master with Persistent World State
"""

from pathlib import Path
import gradio as gr
from src.agents.game_master import GameMasterAgent
from src.llm import get_llm
from src.world.state import WorldState
from src.image.flux import generate_scene_image

# # TODO: remove
# def generate_scene_image(prompt):
#     return None

LLM_NAME = "gemini"  # or "claude"


### Game logic ########################################

def load_default_world() -> WorldState:
    with open("data/worlds/default.json") as f:
        return WorldState.from_json(f.read())


def start_game(world_state_json: str) -> tuple:
    """start a new session. Loads the world and generates the opening scene."""
    state = load_default_world()
    llm = get_llm(LLM_NAME)
    gm = GameMasterAgent(llm)

    opening = gm.generate_opening(state)
    narration = opening["narration"]
    image = None

    if opening.get("image_prompt"):
        try:
            image = generate_scene_image(opening["image_prompt"])
        except Exception:
            pass

    chat_history = [{"role": "assistant", "content": narration}]
    return chat_history, image, state.to_json(), state.to_player_summary()


def take_action(
    action: str,
    chat_history: list,
    world_state_json: str,
    current_image,
) -> tuple:
    """game loop that process one player turn and return updated state."""
    if not action.strip():
        return chat_history, current_image, world_state_json, ""

    state = WorldState.from_json(world_state_json)
    llm = get_llm(LLM_NAME)
    gm = GameMasterAgent(llm)

    result = gm.process_turn(action, state)

    chat_history = chat_history + [
        {"role": "user",      "content": action},
        {"role": "assistant", "content": result["narration"]},
    ]

    new_image = current_image
    if result["scene_changed"] and result.get("image_prompt"):
        try:
            new_image = generate_scene_image(result["image_prompt"])
        except Exception:
            pass

    return chat_history, new_image, state.to_json(), state.to_player_summary()


### Theme toggle
TOGGLE_JS = """
() => {
    document.body.classList.toggle('light-mode');
}
"""

# Ensure dark mode is the default load state
LOAD_JS = """
() => {
    document.body.classList.remove('light-mode');
}
"""

def toggle_theme(is_light):
    is_light = not is_light
    label = "🌑 DARK" if is_light else "☀ LIGHT"
    return is_light, gr.update(value=label)


### Gradio UI ########################################

with gr.Blocks(title="Quilltale") as demo:

    world_state = gr.State("")
    theme_state = gr.State(False)

    # Top bar
    with gr.Row(elem_classes=["qt-topbar"]):
        gr.HTML("""
            <div class="qt-brand">
                <span class="qt-rune">᚛</span>
                <span class="qt-brand-name">Quilltale</span>
                <span class="qt-rune">᚜</span>
            </div>
        """)
        theme_btn = gr.Button(
            "☀  LIGHT",
            variant="secondary",
            scale=0,
            min_width=110,
            elem_classes=["qt-toggle-btn"],
        )

    # Tagline
    gr.HTML("""
        <p class="qt-tagline">
            An AI Game Master with persistent RPG world state, memory driven NPCs, and evolving storytelling.
        </p>
    """)

    # Main layout
    with gr.Row(elem_classes=["qt-main-row"]):

        # Left Column
        with gr.Column(scale=5, elem_classes=["qt-left-col"]):

            gr.HTML('<div class="qt-panel-label">CURRENT SCENE</div>')
            scene_image = gr.Image(
                label="",
                interactive=False,
                show_label=False,
                height=280,
                elem_classes=["qt-scene"],
            )

            with gr.Accordion("WORLD SNAPSHOT", open=False, elem_classes=["qt-accord"]):
                world_state_display = gr.Textbox(
                    label="",
                    interactive=False,
                    lines=14,
                    elem_classes=["qt-world-text"],
                )

            new_game_btn = gr.Button(
                "↺  Start Over",
                variant="secondary",
                elem_classes=["qt-btn-new"],
            )

        # Right Column
        with gr.Column(scale=7, elem_classes=["qt-right-col"]):

            gr.HTML('<div class="qt-panel-label">THE CHRONICLE</div>')
            chatbot = gr.Chatbot(
                label="",
                show_label=False,
                height=480,
                elem_classes=["qt-chronicle"],
            )

            gr.HTML('<div class="qt-input-label">What happens next in your adventure?</div>')
            with gr.Row(elem_classes=["qt-input-row"]):
                action_input = gr.Textbox(
                    placeholder="You are a stranger who has just arrived in a dark medieval city called The Ashen Reach. You wake up in a tavern with an old iron key in your pocket that you don't remember acquiring.Say what you want to do next... look around, talk to Marta, grab the dagger, head north...",
                    label="",
                    scale=5,
                    lines=3,
                    max_lines=3,
                    elem_classes=["qt-action-box"],
                )
                submit_btn = gr.Button(
                    "ACT",
                    variant="primary",
                    scale=1,
                    min_width=50,
                    interactive=False,
                    elem_classes=["qt-btn-act"],
                )

    # Footer
    gr.HTML("""
        <div class="qt-footer">
            <span class="qt-footer-rule">— ✦ —</span>
            <span class="qt-footer-text">
                The story changes with every decision you make, and NPCs remember your actions.
            </span>
            <span class="qt-footer-rule">— ✦ —</span>
        </div>
    """)


    ### Event wiring ########################################

    theme_btn.click(
        fn=toggle_theme,
        inputs=[theme_state],
        outputs=[theme_state, theme_btn],
        js=TOGGLE_JS
    )

    new_game_btn.click(
        fn=start_game,
        inputs=[world_state],
        outputs=[chatbot, scene_image, world_state, world_state_display],
    )

    submit_btn.click(
        fn=lambda: (gr.update(interactive=False), gr.update(interactive=False)),
        outputs=[submit_btn, action_input],
    ).then(
        fn=take_action,
        inputs=[action_input, chatbot, world_state, scene_image],
        outputs=[chatbot, scene_image, world_state, world_state_display],
    ).then(
        fn=lambda: ("", gr.update(interactive=True), gr.update(interactive=False)),
        outputs=[action_input, action_input, submit_btn],
    )

    action_input.submit(
        fn=lambda: (gr.update(interactive=False), gr.update(interactive=False)),
        outputs=[submit_btn, action_input],
    ).then(
        fn=take_action,
        inputs=[action_input, chatbot, world_state, scene_image],
        outputs=[chatbot, scene_image, world_state, world_state_display],
    ).then(
        fn=lambda: ("", gr.update(interactive=True), gr.update(interactive=False)),
        outputs=[action_input, action_input, submit_btn],
    )

    action_input.change(
        fn=lambda text: gr.update(interactive=bool(text.strip())),
        inputs=[action_input],
        outputs=[submit_btn],
    )

    demo.load(
        fn=start_game,
        inputs=[world_state],
        outputs=[chatbot, scene_image, world_state, world_state_display],
        js=LOAD_JS,
    )


if __name__ == "__main__":
    demo.launch(
        css_paths=Path("assets/styles.css"),
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )

