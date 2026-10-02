"""Inject the language controls into the offline Gradio shell."""

from pathlib import Path

import gradio as gr
import kohya_gui.localization as localization

_ROOT = Path(__file__).resolve().parent.parent


def js_html_str(language=None):
    script = (_ROOT / "assets" / "js" / "localization.js").read_text(encoding="utf-8")
    return (
        '<script type="text/javascript">'
        + localization.load_language_js(language)
        + script
        + "</script>\n"
    )


def add_javascript(language=None):
    # Both languages are always available, including launches without --language.
    js = js_html_str(language).encode("utf-8")

    def template_response(*args, **kwargs):
        res = localization.GrRoutesTemplateResponse(*args, **kwargs)
        res.body = res.body.replace(b"</head>", js + b"</head>", 1)
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


if not hasattr(localization, "GrRoutesTemplateResponse"):
    localization.GrRoutesTemplateResponse = gr.routes.templates.TemplateResponse
