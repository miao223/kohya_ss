/* Bilingual display text only: input values, prompts, filenames and code stay intact. */
(() => {
    "use strict";
    const options = window.kohyaI18n;
    if (!options) return;
    const storageKey = "kohya-ui-language";
    const translations = options.translations;
    const normalize = (text) => text.trim().replace(/\s+/g, " ");
    const dictionary = new Map(Object.entries(translations).map(([key, value]) => [normalize(key), value]));
    const originals = new WeakMap();
    const attributes = new WeakMap();
    // Gradio's own chrome can initially follow the browser's system locale.
    const chromeEnglish = new Map([
        ["设置", "Settings"], ["语言", "Language"], ["显示主题", "Display Theme"],
        ["加载中", "Loading"], ["加载中...", "Loading..."], ["使用 Gradio 构建", "Built with Gradio"],
        ["通过 API 使用", "Use via API"], ["标志", "logo"],
        ["渐进式 Web 应用", "Progressive Web App"],
    ]);
    let language = options.defaultLanguage;
    try {
        const saved = localStorage.getItem(storageKey);
        if (saved === "en" || saved === "zh-CN") language = saved;
    } catch (_) { /* Private browsing can disable storage. */ }
    let observer;
    let root;
    let switcher;
    const ignored = "script, style, textarea, code, pre, [contenteditable], #kohya-language-switcher, .kohya-language-setting-hidden";
    const attributeIgnored = "script, style, [contenteditable], #kohya-language-switcher, .kohya-language-setting-hidden";

    function display(source) {
        const english = (chromeEnglish.get(normalize(source)) || source)
            .replaceAll("开始录制", "Start Recording").replaceAll("停止录制", "Stop Recording");
        if (language === "en") return english;
        for (const [prefix, label] of [["number input for ", "数值输入："], ["range slider for ", "滑块："]]) {
            if (english.startsWith(prefix)) {
                const field = english.slice(prefix.length);
                return label + (dictionary.get(normalize(field)) || field);
            }
        }
        let translated = dictionary.get(normalize(english));
        if (translated === undefined) {
            const icon = english.match(/^([^\p{L}\p{N}]+)([\p{L}\p{N}].*)$/u);
            if (icon && dictionary.has(normalize(icon[2]))) {
                return icon[1] + dictionary.get(normalize(icon[2]));
            }
        }
        if (translated === undefined) return english;
        // Keep whitespace separating inline elements and icon text.
        const leading = english.match(/^\s*/)[0];
        const trailing = english.match(/\s*$/)[0];
        return leading + translated + trailing;
    }

    function text(node) {
        if (!node.parentElement || node.parentElement.closest(ignored)) return;
        const current = node.nodeValue;
        if (!current.trim()) return;
        let record = originals.get(node);
        // A framework update may reuse the node with new source text.
        if (!record || (current !== record.last && current !== record.source)) {
            record = { source: current, last: current };
            originals.set(node, record);
        }
        const next = display(record.source);
        if (current !== next) node.nodeValue = next;
        record.last = next;
    }

    function element(node) {
        if (node.matches(attributeIgnored) || node.closest(attributeIgnored)) return;
        let records = attributes.get(node);
        if (!records) {
            records = new Map();
            attributes.set(node, records);
        }
        for (const name of ["title", "placeholder", "aria-label", "alt"]) {
            if (!node.hasAttribute(name)) continue;
            const current = node.getAttribute(name);
            let record = records.get(name);
            if (!record || (current !== record.last && current !== record.source)) {
                record = { source: current, last: current };
                records.set(name, record);
            }
            const next = display(record.source);
            if (current !== next) node.setAttribute(name, next);
            record.last = next;
        }
    }

    function translateTree(node) {
        if (node.nodeType === Node.TEXT_NODE) { text(node); return; }
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        element(node);
        const walker = document.createTreeWalker(node, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
        while (walker.nextNode()) {
            if (walker.currentNode.nodeType === Node.TEXT_NODE) text(walker.currentNode);
            else element(walker.currentNode);
        }
    }

    function hideLanguageSetting() {
        // Hide only Gradio's language section, keeping theme and other settings.
        root.querySelectorAll(".banner-wrap").forEach((section) => {
            const heading = section.querySelector("h2");
            if (!heading) return;
            const label = normalize(heading.textContent);
            if (label === "Language" || label === "语言") {
                section.classList.add("kohya-language-setting-hidden");
                section.hidden = true;
            }
        });
    }

    function updateButtons() {
        switcher.querySelectorAll("button").forEach((button) => {
            button.setAttribute("aria-pressed", String(button.dataset.language === language));
        });
        document.documentElement.lang = language;
    }

    function mountSwitcher() {
        const container = root.querySelector(".gradio-container");
        if (!container) return false;
        if (!switcher || !switcher.isConnected) {
            switcher = document.createElement("div");
            switcher.id = "kohya-language-switcher";
            switcher.setAttribute("role", "group");
            switcher.setAttribute("aria-label", "中文 / English");
            for (const [locale, label] of [["zh-CN", "中文"], ["en", "English"]]) {
                const button = document.createElement("button");
                button.type = "button";
                button.textContent = label;
                button.dataset.language = locale;
                button.addEventListener("click", () => {
                    language = locale;
                    try { localStorage.setItem(storageKey, locale); } catch (_) { /* No persistence. */ }
                    // No navigation or server calls: form values and tab selection survive.
                    refresh(() => translateTree(root));
                    updateButtons();
                });
                switcher.append(button);
            }
            container.prepend(switcher);
        }
        updateButtons();
        return true;
    }

    function watch() {
        observer.observe(root, { subtree: true, childList: true, characterData: true, attributes: true,
            attributeFilter: ["title", "placeholder", "aria-label", "alt"] });
    }

    function refresh(action) {
        // Disconnect while writing translations to avoid observer feedback loops.
        observer.disconnect();
        try {
            mountSwitcher();
            hideLanguageSetting();
            action();
        } finally { watch(); }
    }

    function start() {
        const app = document.querySelector("gradio-app");
        root = app?.shadowRoot || document.body;
        observer = new MutationObserver((mutations) => {
            refresh(() => {
                for (const mutation of mutations) {
                    if (mutation.type === "childList") mutation.addedNodes.forEach(translateTree);
                    else if (mutation.type === "characterData") text(mutation.target);
                    else element(mutation.target);
                }
            });
        });
        refresh(() => translateTree(root));
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
    else start();
})();
