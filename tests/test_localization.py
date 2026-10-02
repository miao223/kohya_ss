"""Bilingual catalogue coverage, safe injection, and Markdown fragmentation."""

import ast
import json
from pathlib import Path
from unittest.mock import patch

from kohya_gui.localization import load_catalogue, load_language_js, translation_map
from kohya_gui.localization_ext import js_html_str

ROOT = Path(__file__).resolve().parents[1]


def test_catalogue_can_load_outside_project_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_catalogue()["Epoch"] == "训练轮数"
    assert "kohyaI18n" in js_html_str()


def test_language_default_and_safe_script_payload():
    for language, expected in [
        (None, "zh-CN"),
        ("zh-CN", "zh-CN"),
        ("en", "en"),
        ("en-GB", "en"),
    ]:
        with patch(
            "kohya_gui.localization.translation_map",
            return_value={"X": "</script><script>alert(1)</script>"},
        ):
            script = load_language_js(language)
        assert "</script>" not in script
        payload = json.loads(
            script.removeprefix("window.kohyaI18n = ").removesuffix(";")
        )
        assert payload["defaultLanguage"] == expected
        assert payload["translations"]["X"].startswith("</script>")


def test_markdown_fragments_have_translations():
    translations = translation_map()
    for fragment in [
        "ONNX (recommended, default):",
        "Prefix / postfix",
        "Train an Anima",
        "Experimental:",
        "Access EDG's tutorials here:",
    ]:
        assert fragment in translations
        assert translations[fragment] != fragment
    assert translations["Max resolution"] == "最大分辨率"


def test_all_static_field_labels_and_help_are_covered():
    catalogue = load_catalogue()
    # Brand names, a numeric resolution, and a copyable example prompt are
    # deliberately retained. Do not translate training values or code.
    retained = {
        "SDXL",
        "SD3",
        "Flux.1",
        "HunyuanImage-2.1",
        "Anima",
        "Lumina",
        "512,512",
    }
    missing = []
    for path in (ROOT / "kohya_gui").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg not in {"label", "info", "placeholder"}:
                    continue
                value = keyword.value
                if not isinstance(value, ast.Constant) or not isinstance(
                    value.value, str
                ):
                    continue
                source = value.value
                if (
                    source.strip()
                    and source not in retained
                    and not source.startswith("masterpiece, best quality,")
                    and source not in catalogue
                ):
                    missing.append((path.name, value.lineno, source))
    assert not missing, missing


def test_training_identifiers_stay_in_english():
    catalogue = load_catalogue()
    for identifier in [
        "Prodigy",
        "PagedAdamW8bit",
        "constant",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
    ]:
        assert catalogue[identifier] == identifier
    assert catalogue["Gradient checkpointing"] == "梯度检查点"
    assert catalogue["Gradient accumulate steps"] == "梯度累积步数"
