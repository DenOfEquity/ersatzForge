// mouseover tooltips for various UI elements
// mostly useless

var titles = {
    "Sampling steps": "How many times to improve the generated image iteratively; higher values take longer; very low values can produce bad results",
    "Sampling method": "Which algorithm to use to produce the image",
    "GFPGAN": "Restore low quality faces using GFPGAN neural network",

    "\u{1F4D0}": "Auto detect size from img2img",
    "Batch count": "How many batches of images to create (has no impact on generation performance or VRAM usage)",
    "Batch size": "How many image to create in a single batch (increases generation performance at cost of higher VRAM usage)",
    "CFG Scale": "Classifier Free Guidance Scale - how strongly the image should conform to prompt - lower values produce more creative results",
    "Seed": "A value that determines the output of random number generator - if you create an image with same parameters and seed as another image, you'll get the same result",
    "\u{1f3b2}\ufe0f": "Set seed to -1, which will cause a new random number to be used every time",
    "\u267b\ufe0f": "Reuse seed from last generation, mostly useful if it was randomized",
    "\u2199\ufe0f": "Read generation parameters from prompt or last generation if prompt is empty into user interface.",
    "\u{1f4c2}": "Open images output directory",
    "\u{1f4be}": "Save style",
    "\u{1f5d1}\ufe0f": "Clear prompt",
    "\u{1f4cb}": "Apply selected styles to current prompt",
    "\u{1f4d2}": "Paste available values into the field",
    "\u{1f3b4}": "Show/hide extra networks",
    "\u{1f300}": "Restore progress",

    "Inpaint a part of image": "Draw a mask over an image, and the script will regenerate the masked area with content according to prompt",
    "SD upscale": "Upscale image normally, split result into tiles, improve each tile using img2img, merge whole image back",

    "Denoising strength": "Determines how little respect the algorithm should have for image's content. At 0, nothing will change, and at 1 you'll get an unrelated image. With values below 1.0, processing will take less steps than the Sampling Steps slider specifies.",

    "Skip": "Stop processing current image and continue processing.",
    "Interrupt": "Stop processing images and return any results accumulated so far.",

    "X values": "Separate values for X axis using commas.",
    "Y values": "Separate values for Y axis using commas.",

    "None": "Do not do anything special",
    "Prompt matrix": "Separate prompts into parts using vertical pipe character (|) and the script will create a picture for every combination of them (except for the first part, which will be present in all combinations)",
    "X/Y/Z plot": "Create grid(s) where images will have different parameters. Use inputs below to specify which parameters will be shared by columns and rows",
    "Custom code": "Run Python code. Advanced user only. Must run program with --allow-code for this to work",

    "Prompt S/R": "Separate a list of words with commas, and the first word will be used as a keyword: script will search for this word in the prompt, and replace it with others",
    "Prompt order": "Separate a list of words with commas, and the script will make a variation of prompt with those words for their every possible order",

    "Tile overlap": "For SD upscale, how much overlap in pixels should there be between tiles. Tiles overlap so that when they are merged back into one picture, there is no clearly visible seam.",

    "Interrogate": "Reconstruct prompt from existing image and put it into the prompt field.",

    "Images filename pattern": "Use tags like [seed] and [date] to define how filenames for images are chosen. Leave empty for default.",
    "Directory name pattern": "Use tags like [seed] and [date] to define how subdirectories for images and grids are chosen. Leave empty for default.",
    "Max prompt words": "Set the maximum number of words to be used in the [prompt_words] option; ATTENTION: If the words are too long, they may exceed the maximum length of the file path that the system can handle",

    "Loopback": "Performs img2img processing multiple times. Output images are used as input for the next loop.",
    "Loops": "How many times to process an image. Each output is used as the input of the next loop. If set to 1, behavior will be as if this script were not used.",
    "Final denoising strength": "The denoising strength for the final loop of each image in the batch.",
    "Denoising strength curve": "The denoising curve controls the rate of denoising strength change each loop. Aggressive: Most of the change will happen towards the start of the loops. Linear: Change will be constant through all loops. Lazy: Most of the change will happen towards the end of the loops.",

    "Style 1": "Style to apply; styles have components for both positive and negative prompts and apply to both",
    "Style 2": "Style to apply; styles have components for both positive and negative prompts and apply to both",
    "Apply style": "Insert selected styles into prompt fields",
    "Create style": "Save current prompts as a style. If you add the token {prompt} to the text, the style uses that as a placeholder for your prompt when you use the style in the future.",

    "Checkpoint name": "Loads weights from checkpoint before making images. You can either use hash or a part of filename (as seen in settings) for checkpoint name. Recommended to use with Y axis for less switching.",
    "Inpainting conditioning mask strength": "Only applies to inpainting models. Determines how strongly to mask off the original image for inpainting and img2img. 1.0 means fully masked, which is the default behaviour. 0.0 means a fully unmasked conditioning. Lower values will help preserve the overall composition of the image, but will struggle with large changes.",

    "Eta noise seed delta": "If this values is non-zero, it will be added to seed and used to initialize RNG for noises when using samplers with Eta. You can use this to produce even more variation of images, or you can use this to match images of other software if you know what you are doing.",

    "Quicksettings list": "List of setting names, separated by commas, for settings that should go to the quick access bar at the top, rather than the usual setting tab. See modules/shared.py for setting names. Requires restarting to apply.",

    "Extra networks tab order": "Comma-separated list of tab names; tabs listed here will appear in the extra networks UI first and in order listed.",
    "Negative Guidance minimum sigma": "Skip negative prompt for steps where image is already mostly denoised; the higher this value, the more skips there will be; provides increased performance in exchange for minor quality reduction."
};

function updateTooltip(element) {
    if (element.title) return; // already has a title

    let text = element.textContent;
    let tooltip = localization[titles[text]] || titles[text];

    if (!tooltip) {
        let value = element.value;
        if (value) tooltip = localization[titles[value]] || titles[value];
    }

    if (!tooltip) {
        // Gradio dropdown options have `data-value`.
        let dataValue = element.dataset.value;
        if (dataValue) tooltip = localization[titles[dataValue]] || titles[dataValue];
    }

    if (!tooltip) {
        for (const c of element.classList) {
            if (c in titles) {
                tooltip = localization[titles[c]] || titles[c];
                break;
            }
        }
    }

    if (tooltip) {
        element.title = tooltip;
    }
}

// Nodes to check for adding tooltips.
const tooltipCheckNodes = new Set();
// Timer for debouncing tooltip check.
let tooltipCheckTimer = null;

function processTooltipCheckNodes() {
    for (const node of tooltipCheckNodes) {
        updateTooltip(node);
    }
    tooltipCheckNodes.clear();
}

/*
onUiUpdate(function(mutationRecords) {
    for (const record of mutationRecords) {
        if (record.type === "childList" && record.target.classList.contains("options")) {
            // This smells like a Gradio dropdown menu having changed,
            // so let's enqueue an update for the input element that shows the current value.
            let wrap = record.target.parentNode;
            let input = wrap?.querySelector("input");
            if (input) {
                input.title = ""; // So we'll even have a chance to update it.
                tooltipCheckNodes.add(input);
            }
        }
        for (const node of record.addedNodes) {
            if (node.nodeType === Node.ELEMENT_NODE && !node.classList.contains("hide")) {
                if (!node.title) {
                    if (
                        node.tagName === "SPAN" ||
                        node.tagName === "BUTTON" ||
                        node.tagName === "P" ||
                        node.tagName === "INPUT" ||
                        (node.tagName === "LI" && node.classList.contains("item")) // Gradio dropdown item
                    ) {
                        tooltipCheckNodes.add(node);
                    }
                }
                node.querySelectorAll('span, button, p').forEach(n => tooltipCheckNodes.add(n));
            }
        }
    }
    // if (tooltipCheckNodes.size) {
        // clearTimeout(tooltipCheckTimer);
        // tooltipCheckTimer = setTimeout(processTooltipCheckNodes, 1000);
    // }
});
*/
onUiLoaded(function() {
    for (var comp of window.gradio_config.components) {
        if (comp.props.webui_tooltip && comp.props.elem_id) {
            var elem = gradioApp().getElementById(comp.props.elem_id);
            if (elem) {
                elem.title = comp.props.webui_tooltip;
            }
        }
    }
});
