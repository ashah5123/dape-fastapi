"""
Gradio UI for the DAPE FastAPI Docs Assistant.

Exposes a simple text generation interface on top of the existing FastAPI API.
"""

from typing import Callable

import gradio as gr


def build_ui(generate_fn: Callable[[str, int], str]) -> gr.Blocks:
    """
    Build and return a Gradio Blocks UI.

    The `generate_fn` callable should accept (prompt: str, max_new_tokens: int)
    and return a string response. Any exceptions are caught and rendered as
    user-friendly error messages.
    """

    def _on_generate(prompt: str, max_new_tokens: int) -> str:
        if not prompt.strip():
            return "Please provide a non-empty prompt."
        try:
            return generate_fn(prompt, max_new_tokens)
        except Exception as e:  # noqa: BLE001
            # In particular, handle the \"model not loaded\" case gracefully.
            msg = str(e)
            if "Model not loaded" in msg:
                return "Model is not yet loaded. Please try again in a few seconds."
            return f"Error generating response: {msg}"

    with gr.Blocks() as demo:
        gr.Markdown("## DAPE — FastAPI Docs Assistant")

        with gr.Row():
            prompt_in = gr.Textbox(
                label="Prompt",
                placeholder="Ask a FastAPI question, e.g. 'How do I define a GET endpoint?'",
                lines=4,
            )

        max_tokens_in = gr.Slider(
            minimum=16,
            maximum=256,
            value=120,
            step=8,
            label="Max new tokens",
        )

        generate_btn = gr.Button("Generate")

        output_box = gr.Textbox(
            label="Output",
            lines=10,
        )

        generate_btn.click(
            fn=_on_generate,
            inputs=[prompt_in, max_tokens_in],
            outputs=[output_box],
        )

    return demo

