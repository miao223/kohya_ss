/* Workflow links open their section before scrolling; training stays explicit. */
(() => {
    document.addEventListener("click", (event) => {
        const link = event.target.closest?.(".studio-workflow a");
        if (!link) return;
        const target = document.getElementById(link.getAttribute("href").slice(1));
        if (!target) return;
        event.preventDefault();
        const heading = target.querySelector(":scope > button.label-wrap");
        if (heading && !heading.classList.contains("open")) heading.click();
        requestAnimationFrame(() => {
            target.scrollIntoView({ behavior: "auto", block: "start" });
            // Focus the section heading for keyboard users without starting training.
            if (heading) heading.focus({ preventScroll: true });
        });
    });
})();
