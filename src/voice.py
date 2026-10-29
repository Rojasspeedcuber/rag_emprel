import re
from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent / "voice_component"
_component = components.declare_component("voice_control", path=str(_COMPONENT_DIR))


def to_speech_text(markdown: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", markdown)
    text = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`#>]+", "", text).replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def voice_input(key: str, disabled: bool = False) -> dict | None:
    """Botão de ditado. Retorna {"text", "nonce"} quando uma fala é captada."""
    return _component(mode="input", disabled=disabled, key=key, default=None)


def speak_button(text: str, key: str, autoplay: bool = False) -> None:
    """Botões para ouvir e parar a leitura de uma resposta."""
    _component(
        mode="speak",
        text=to_speech_text(text),
        autoplay=autoplay,
        key=key,
        default=None,
    )
