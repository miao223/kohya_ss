/* Merge Gradio's live preferences into the sole Settings tab. */
(() => {
    "use strict";
    function start() {
        const app = document.querySelector("gradio-app");
        const root = app?.shadowRoot || document.body;
        let requested = false;
        let opening = false;
        let scheduled = false;
        let initializing = false;
        let initialSettingsLink = new URL(window.location.href).searchParams.get("view") === "settings";

        function sync() {
            scheduled = false;
            if (initializing) return;
            const host = root.querySelector("#studio-native-settings");
            const tab = root.querySelector("#studio-settings-button");
            const launcher = root.querySelector("footer button.settings");
            if (!tab || !launcher) return;
            document.documentElement.classList.add("kohya-unified-settings");
            if (initialSettingsLink) {
                initialSettingsLink = false;
                initializing = true;
                requested = true;
                tab.click();
                // On a direct settings link, Gradio marks the overlay open but
                // hasn't loaded its component. Its first toggle loads and closes
                // it; wait for that native route update before opening it inline.
                launcher.click();
                function afterClose() {
                    if (new URL(window.location.href).searchParams.get("view") === "settings") {
                        requestAnimationFrame(afterClose);
                        return;
                    }
                    initializing = false;
                    schedule();
                }
                requestAnimationFrame(afterClose);
                return;
            }
            if (!host) return;

            // Match settings controls, never Gradio's API documentation dialog.
            const panel = Array.from(root.querySelectorAll(".api-docs")).find(
                (node) => node.querySelector(".theme-buttons, .record-button")
            );
            if (panel) {
                opening = false;
                if (!host.contains(panel)) {
                    // The original component stays mounted with all native events.
                    // Move the whole panel so its internal DOM relationships survive.
                    panel.classList.add("studio-native-preferences");
                    panel.setAttribute("role", "region");
                    panel.removeAttribute("aria-modal");
                    host.append(panel);
                    if (!requested) tab.click(); // Also handle old ?view=settings links.
                }
            } else if (requested && !opening && tab.getAttribute("aria-selected") === "true") {
                opening = true;
                launcher.click();
            }
        }

        function schedule() {
            if (scheduled) return;
            scheduled = true;
            requestAnimationFrame(sync);
        }

        root.addEventListener("click", (event) => {
            if (event.target.closest?.("#studio-settings-button")) {
                requested = true;
                schedule();
            } else if (event.target.closest?.("#studio-tabs > .tab-wrapper button, #studio-native-settings .record-button")) {
                // Recording closes the native component by design. Do not reopen
                // it until the user returns to Settings.
                requested = false;
            }
        });
        new MutationObserver((mutations) => {
            // Tab selection can also come from keyboard navigation.
            if (mutations.some((mutation) => mutation.type === "attributes"
                && mutation.target.id === "studio-settings-button"
                && mutation.target.getAttribute("aria-selected") === "true")) {
                requested = true;
            }
            schedule();
        }).observe(root, {
            childList: true, subtree: true, attributes: true,
            attributeFilter: ["aria-selected"],
        });
        sync();
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
    else start();
})();
