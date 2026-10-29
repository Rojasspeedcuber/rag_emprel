from pathlib import Path

from src import voice


def test_to_speech_text_removes_markdown_markers():
    text = "**Prazo**: 30 dias.\n\n- item _um_\n- `código`"

    assert voice.to_speech_text(text) == "Prazo: 30 dias. item um código"


def test_to_speech_text_keeps_link_label_only():
    assert voice.to_speech_text("Veja [o guia](https://exemplo.com)") == "Veja o guia"


def test_to_speech_text_collapses_whitespace():
    assert voice.to_speech_text("  linha\n\n\n outra  ") == "linha outra"


def test_voice_component_assets_exist():
    component_dir = Path(voice.__file__).parent / "voice_component"

    assert (component_dir / "index.html").is_file()
