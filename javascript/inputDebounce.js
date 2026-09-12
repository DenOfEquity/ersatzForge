// based on ForgeNeo PR 1449 by NeiroNext + Claude Code
// modified to :
//   target all typeable textareas and inputs instead of pre-set list of elements
//   hardcoded delay, doesn't need to be an option
//   removed some event listeners, seemed unnecessary

function deBounceAll() {;}

(function () {
    /**
     * Gradio re-evaluates its whole component tree on every value change, which costs ~75 ms
     * per keystroke in a build this size, regardless of which field is edited or whether it is
     * even visible. Typing in a prompt therefore runs at ~11 fps.
     *
     * Only the `input` event is withheld from the listeners below, and only until the typing
     * pauses; `keydown`, `keyup`, `paste`, `focus` and `blur` are never touched, and the value
     * of the textarea itself is always current, so anything reading it directly still sees
     * every character. The pending value is flushed before it could be read by the backend:
     * on any pointer press
     */

    /** @type {Map<HTMLTextAreaElement, number>} */
    const pending = new Map();
    /** @type {WeakMap<HTMLTextAreaElement, InputEventInit>} */
    const lastInput = new WeakMap();
    /** @type {Set<HTMLTextAreaElement>} */
    let targets = new Set();
    let delay = 160;

    /** @param {HTMLTextAreaElement} textarea */
    function flush(textarea) {
        const timer = pending.get(textarea);
        if (timer === undefined) return;

        clearTimeout(timer);
        pending.delete(textarea);

        // re-emit with the fields of the last real keystroke: listeners such as tag autocompletion
        // ignore `input` events without an `inputType` (that is how they skip programmatic updates)
        const init = lastInput.get(textarea);
        const event = init?.inputType ? new InputEvent("input", { bubbles: true, ...init }) : new Event("input", { bubbles: true });

        textarea.dataset.debouncedInput = "1";
        textarea.dispatchEvent(event);
    }

    function flushAll() {
        for (const textarea of Array.from(pending.keys())) flush(textarea);
    }

    function onInput(event) {
        const textarea = event.target;
        if (!targets.has(textarea)) return;

        if (textarea.dataset.debouncedInput) {
            // the event this module dispatched itself; let every listener handle it
            delete textarea.dataset.debouncedInput;
            return;
        }

        if (!event.inputType) {
            // programmatic edit (`updateInput()`: Ctrl+Up/Down, undo, extra networks cards...):
            // one event per action, let it through; it also carries any keystrokes still pending
            const timer = pending.get(textarea);
            if (timer !== undefined) clearTimeout(timer);
            pending.delete(textarea);
            return;
        }

        event.stopPropagation();
        lastInput.set(textarea, { inputType: event.inputType, data: event.data, isComposing: event.isComposing });

        const timer = pending.get(textarea);
        if (timer !== undefined) clearTimeout(timer);
        pending.set(textarea, setTimeout(() => flush(textarea), delay));
    }

    function setup() {
        // target all textareas and inputs that can be typed into
        targets = new Set(gradioApp().querySelectorAll("textarea:not([readonly]), input[type='text']:not([readonly]), input[role='listbox']:not([readonly])"));//, input[type='number']:not([readonly])"));
        targets = new Set([...targets].filter(el => !el.disabled && !el.readOnly));
        if (targets.size === 0) return;

//        console.log(targets);

        document.addEventListener("input", onInput, true);
        document.addEventListener("pointerdown", flushAll, true);

        // standard Ctrl+Enter handler (in script.js) modified to call deBounceAll before sending click to Generate button
        deBounceAll = flushAll;
    }

    onUiLoaded(() => {
        setup();
    });
})();

