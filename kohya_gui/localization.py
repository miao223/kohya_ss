"""Load the bilingual GUI catalogue without depending on the working directory."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

from markdown_it import MarkdownIt

_ROOT = Path(__file__).resolve().parent.parent
localizationMap = {}


def load_localizations():
    localizationMap.clear()
    for path in (_ROOT / "localizations").glob("*.json"):
        localizationMap[path.stem] = str(path)


def load_catalogue(language_name="zh-CN"):
    path = localizationMap.get(language_name)
    if path is None:
        return {}
    with open(path, encoding="utf-8") as file:
        return json.load(file)


class _TextSegments(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.segments = []

    def handle_data(self, data):
        # Gradio linkifies bare URLs; mirror its resulting text-node boundaries.
        for segment in re.split(r"(https?://[^\s<>]+)", data):
            if segment.strip():
                self.segments.append(segment.strip())


def translation_map():
    """Include Markdown text fragments while preserving links and inline code.

    The browser translates individual text nodes. Splitting matching Markdown
    structures here lets it translate text around bold text/code/links too.
    Code and URLs are never translated. Mismatched structures use the complete
    message only rather than pairing unrelated fragments.
    """
    catalogue = load_catalogue()
    translations = {key.strip(): value for key, value in catalogue.items()}
    renderer = MarkdownIt("commonmark")
    for source, target in catalogue.items():
        original, translated = _TextSegments(), _TextSegments()
        original.feed(renderer.render(source))
        translated.feed(renderer.render(target))
        if len(original.segments) == len(translated.segments):
            for key, value in zip(original.segments, translated.segments):
                if key != value:
                    translations.setdefault(key, value)
    return translations


def load_language_js(language_name=None):
    # Browser preference wins; --language remains a first-visit default.
    default = "en" if language_name in {"en", "en-GB", "en-US"} else "zh-CN"
    payload = {"defaultLanguage": default, "translations": translation_map()}
    # Do not allow catalogue strings to terminate the inline script element.
    encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    return f"window.kohyaI18n = {encoded};"


load_localizations()
