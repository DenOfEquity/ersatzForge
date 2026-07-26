
function checkBrackets(textArea, counterElem) {
// Stable Diffusion WebUI - Bracket Checker
// By @Bwin4L, @akx, @w-e-w, @Haoming02
// Counts open and closed brackets (round, square, curly) in the prompt and negative prompt text boxes in the txt2img and img2img tabs.
// If there's a mismatch, the keyword counter turns red, and if you hover on it, a tooltip tells you what's wrong.
    const pairs = [
        [0, "(", ")", "round brackets"],
        [1, "<", ">", "angle brackets"],
        [2, "[", "]", "square brackets"],
        [3, "{", "}", "curly brackets"]
    ];

    const counts = [0, 0, 0, 0];
    const errors = new Set();

    let i = 0;
    const length = textArea.value.length - 1;
    while (i <= length) {
        let char = textArea.value[i];
        let escaped = false;
        while (char === "\\" && i < length) {
            escaped = !escaped;
            i++;
            char = textArea.value[i];
        }

        if (!escaped) {
            for (const [index, open, close, label] of pairs) {
                if (char === open) {
                    counts[index] += 1;
                    break;
                }
                else if (char === close) {
                    counts[index] -= 1;
                    if (counts[index] < 0) {
                        errors.add(`Incorrect order of ${label}.`);
                    }
                    break;
                }
            }
        }

        i++;
    }

    for (const [index, open, close, label] of pairs) {
        if (counts[index] > 0) {
            errors.add(`${open} ... ${close} - Detected ${counts[index]} more opening than closing ${label}.`);
        }
        else if (counts[index] < 0) {
            errors.add(`${open} ... ${close} - Detected ${-counts[index]} more closing than opening ${label}.`);
        }
    }

    counterElem.title = [...errors].join("\n");
    counterElem.classList.toggle("error", errors.size !== 0);
}


function setupTokenCounting(id, id_counter, id_button) {
    var prompt   = gradioApp().getElementById(id);
    var counter  = gradioApp().getElementById(id_counter);
    if (counter.parentElement == prompt.parentElement) {
        return;
    }

    var button   = gradioApp().getElementById(id_button);
    var textarea = gradioApp().querySelector("#" + id + " > label > textarea");

    prompt.parentElement.insertBefore(counter, prompt);
    prompt.parentElement.style.position = "relative";

    var func = onEdit(id, textarea, 800, function() {
        checkBrackets(textarea, counter);
        button?.click();
    });
}

onUiLoaded(function() {
    if (!opts.disable_token_counters) {
        setupTokenCounting("txt2img_prompt", "txt2img_token_counter", "txt2img_token_button");
        setupTokenCounting("txt2img_neg_prompt", "txt2img_negative_token_counter", "txt2img_negative_token_button");
        setupTokenCounting("img2img_prompt", "img2img_token_counter", "img2img_token_button");
        setupTokenCounting("img2img_neg_prompt", "img2img_negative_token_counter", "img2img_negative_token_button");
    }
});
