from __future__ import annotations

import os


DEFAULT_MODEL = "gpt-5.2"


def get_model(model: str | None = None) -> str:
    return model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def get_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a key in the OpenAI dashboard, "
            "export it in your shell, then run `python3 eval.py` again."
        )
    return api_key
