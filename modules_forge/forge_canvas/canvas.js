// original (AGPL3): lllyasviel https://github.com/lllyasviel/stable-diffusion-webui-forge
// deobfuscate, hotkeys: Haoming02 https://github.com/Haoming02/sd-webui-forge-classic
// eraser, consolidation of eventlisteners, assorted little adjustments: DoE

class GradioTextAreaBind {
    constructor(id, className) {
        this.target = document.querySelector(`#${id}.${className} textarea`);
        this.sync_lock = false;
        this.previousValue = "";
    }

    set_value(value) {
        if (this.sync_lock) return;
        this.sync_lock = true;
        this.target.value = value;
        this.previousValue = value;
        const event = new Event("input", { bubbles: true });
        Object.defineProperty(event, "target", { value: this.target });
        this.target.dispatchEvent(event);
        this.previousValue = value;
        this.sync_lock = false;
    }

    listen(callback) {
        setInterval(() => {
            if (this.target.value !== this.previousValue) {
                this.previousValue = this.target.value;
                if (this.sync_lock) return;
                this.sync_lock = true;
                callback(this.target.value);
                this.sync_lock = false;
            }
        }, 750);
    }
}

class ForgeCanvas {
    constructor(
        uuid,
        no_upload = false,
        no_scribbles = false,
        contrast_scribbles = false,
        initial_height = 512,
        scribbleColor = "#000000",
        scribbleColorFixed = false,
        scribbleWidth = 4,
        scribbleWidthFixed = false,
        scribbleAlpha = 100,
        scribbleAlphaFixed = false,
        scribbleSoftness = 0,
        scribbleSoftnessFixed = false,
    ) {
        this.gradio_config = gradio_config;
        this.uuid = uuid;

        this.no_upload = no_upload;
        this.no_scribbles = no_scribbles;
        this.contrast_scribbles = contrast_scribbles;

        this.img = null;
        this.imgX = 0;
        this.imgY = 0;
        this.orgWidth = 0;
        this.orgHeight = 0;
        this.imgScale = 1;
        this.initial_height = initial_height;

        this.dragging = false;
        this.dragged_just_now = false;
        this.resizing = false;
        this.drawing = false;
        this.contrast_pattern = null;

        this.scribbleColor = scribbleColor;
        this.scribbleColorFixed = (scribbleColorFixed || this.contrast_scribbles);
        this.scribbleWidth = scribbleWidth;
        this.scribbleWidthFixed = scribbleWidthFixed;
        this.scribbleAlpha = scribbleAlpha;
        this.scribbleAlphaFixed = (scribbleAlphaFixed || this.contrast_scribbles);
        this.scribbleSoftness = scribbleSoftness;
        this.scribbleSoftnessFixed = (scribbleSoftnessFixed || this.contrast_scribbles);

        this.history = [];
        this.historyIndex = -1;
        this.maximized = false;
        this.originalState = {};
        this.pointerInsideContainer = false;
        this.temp_canvas = document.createElement("canvas");
        this.temp_draw_points = [];
        this.temp_draw_bg = null;
        this.erasing = false;

        this.background_gradio_bind = new GradioTextAreaBind(this.uuid, "logical_image_background");
        this.foreground_gradio_bind = new GradioTextAreaBind(this.uuid, "logical_image_foreground");
        this.init();

        this._held_W = false;
        this._held_A = false;
        this._held_S = false;
    }

    init() {
        this.container = document.getElementById(`container_${this.uuid}`);
        const imageContainer = document.getElementById(`imageContainer_${this.uuid}`);
        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const toolbar = document.getElementById(`toolbar_${this.uuid}`);

        this.maxButton = document.getElementById(`maxButton_${this.uuid}`);
        const uploadButton = document.getElementById(`uploadButton_${this.uuid}`);
        const removeButton = document.getElementById(`removeButton_${this.uuid}`);
        const centerButton = document.getElementById(`centerButton_${this.uuid}`);
        const resetButton = document.getElementById(`resetButton_${this.uuid}`);
        const undoButton = document.getElementById(`undoButton_${this.uuid}`);
        const redoButton = document.getElementById(`redoButton_${this.uuid}`);
        this.eraserButton = document.getElementById(`eraserButton_${this.uuid}`);

        const uploadHint = document.getElementById(`uploadHint_${this.uuid}`);
        const scribbleIndicator = document.getElementById(`scribbleIndicator_${this.uuid}`);

        if (this.scribbleColorFixed) document.getElementById(`scribbleColorBlock_${this.uuid}`).style.display = "none";
        const scribbleColor = document.getElementById(`scribbleColor_${this.uuid}`);
        scribbleColor.value = this.scribbleColor;

        if (this.scribbleWidthFixed) document.getElementById(`scribbleWidthBlock_${this.uuid}`).style.display = "none";
        const scribbleWidth = document.getElementById(`scribbleWidth_${this.uuid}`);
        scribbleWidth.value = this.scribbleWidth;

        if (this.scribbleAlphaFixed) document.getElementById(`scribbleAlphaBlock_${this.uuid}`).style.display = "none";
        const scribbleAlpha = document.getElementById(`scribbleAlpha_${this.uuid}`);
        scribbleAlpha.value = this.scribbleAlpha;

        if (this.scribbleSoftnessFixed) document.getElementById(`scribbleSoftnessBlock_${this.uuid}`).style.display = "none";
        const scribbleSoftness = document.getElementById(`scribbleSoftness_${this.uuid}`);
        scribbleSoftness.value = this.scribbleSoftness;

        const indicatorSize = this.scribbleWidth * 20;
        scribbleIndicator.style.width = `${indicatorSize}px`;
        scribbleIndicator.style.height = `${indicatorSize}px`;

        this.container.style.height = `${this.initial_height}px`;
        drawingCanvas.width = imageContainer.clientWidth;
        drawingCanvas.height = imageContainer.clientHeight;

        const drawContext = drawingCanvas.getContext("2d", { willReadFrequently: true });
        drawingCanvas.style.cursor = "crosshair";


        if (this.no_scribbles) {
            toolbar.querySelector(".forge-toolbar-box-b").style.display = "none";
            resetButton.style.display = "none";
            undoButton.style.display = "none";
            redoButton.style.display = "none";
            this.eraserButton.style.display = "none";
        }

        if (this.no_upload) {
            uploadButton.style.display = "none";
            uploadHint.style.display = "none";
        }

        if (this.contrast_scribbles) {
            const size = 10;
            const tempCanvas = this.temp_canvas;
            tempCanvas.width = size * 2;
            tempCanvas.height = size * 2;
            const tempCtx = tempCanvas.getContext("2d");
            tempCtx.fillStyle = "#ffffff";
            tempCtx.fillRect(0, 0, size, size);
            tempCtx.fillRect(size, size, size, size);
            tempCtx.fillStyle = "#000000";
            tempCtx.fillRect(size, 0, size, size);
            tempCtx.fillRect(0, size, size, size);
            this.contrast_pattern = drawContext.createPattern(tempCanvas, "repeat");
            drawingCanvas.style.opacity = "0.5";
        }

        const resizeObserver = new ResizeObserver(() => {
            this.adjustInitialPositionAndScale();
            this.drawImage();
        });
        resizeObserver.observe(this.container);

        document.getElementById(`imageInput_${this.uuid}`).addEventListener("change", (e) => {
            this.handleFileUpload(e.target.files[0]);
            if (this.img && !this.no_scribbles) scribbleIndicator.style.display = "inline-block";
        });

        uploadButton.addEventListener("click", () => {
            if (this.no_upload) return;
            document.getElementById(`imageInput_${this.uuid}`).click();
            if (this.img && !this.no_scribbles) scribbleIndicator.style.display = "inline-block";
        });

        removeButton.addEventListener("click", () => {
            this.removeImage();
            this.resetImage();
            scribbleIndicator.style.display = "none";
        });

        centerButton.addEventListener("click", () => {
            this.adjustInitialPositionAndScale();
            this.drawImage();
        });

        resetButton.addEventListener("click", () => {
            this.resetImage();
        });

        undoButton.addEventListener("click", () => {
            this.undo();
        });

        redoButton.addEventListener("click", () => {
            this.redo();
        });

        scribbleColor.addEventListener("input", (e) => {
            this.scribbleColor = e.target.value;
            scribbleIndicator.style.borderColor = this.scribbleColor;
        });

        scribbleWidth.addEventListener("input", (e) => {
            this.scribbleWidth = e.target.value;
            const indicatorSize = this.scribbleWidth * 20;
            scribbleIndicator.style.width = `${indicatorSize}px`;
            scribbleIndicator.style.height = `${indicatorSize}px`;
        });

        scribbleAlpha.addEventListener("input", (e) => {
            this.scribbleAlpha = e.target.value;
        });

        scribbleSoftness.addEventListener("input", (e) => {
            this.scribbleSoftness = e.target.value;
        });

        this.container.addEventListener("pointerdown", (e) => {
            const rect = this.container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            if (e.button === 2 && this.isInsideImage(x, y)) {
                this.dragging = true;
                this.offsetX = x - this.imgX;
                this.offsetY = y - this.imgY;
                drawingCanvas.style.cursor = "grabbing";
                scribbleIndicator.style.display = "none";
            }
            else if (e.button === 1 && this.img) { // middle-click: center 
                this.adjustInitialPositionAndScale();
                this.drawImage();
                e.preventDefault();
            }
            else if (e.button === 0) {
                if (!this.img && !this.no_upload) {
                    document.getElementById(`imageInput_${this.uuid}`).click();
                }
                else if (this.img && !this.no_scribbles) {
                    const rect = drawingCanvas.getBoundingClientRect();
                    this.drawing = true;
                    drawingCanvas.style.cursor = "crosshair";
                    this.temp_draw_points = [[(e.clientX - rect.left) / this.imgScale, (e.clientY - rect.top) / this.imgScale]];
                    this.temp_draw_bg = drawContext.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height);
                    this.handleDraw(e);
                }
            }
        });

        this.container.addEventListener("pointermove", (e) => {
            if (this.drawing) this.handleDraw(e);
            else if (this.dragging) {
                const rect = this.container.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                this.imgX = x - this.offsetX;
                this.imgY = y - this.offsetY;
                this.drawImage();
                this.dragged_just_now = true;
            }
            if (this.img && !this.dragging && !this.no_scribbles) {
                const rect = this.container.getBoundingClientRect();
                const indicatorSize = this.scribbleWidth * 10;
                scribbleIndicator.style.left = `${e.clientX - rect.left - indicatorSize}px`;
                scribbleIndicator.style.top = `${e.clientY - rect.top - indicatorSize}px`;
            }
        });

        this.container.addEventListener("pointerup", () => {
            if (this.drawing) {
                this.drawing = false;
                this.saveState();
            }
            if (this.dragging) {
                this.dragging = false;
            }
            drawingCanvas.style.cursor = "crosshair";
            if (this.img && !this.no_scribbles) scribbleIndicator.style.display = "inline-block";
        });

        const resizeLine = document.getElementById(`resizeLine_${this.uuid}`);
        resizeLine.addEventListener("pointerdown", (e) => {
            this.resizing = true;
            scribbleIndicator.style.display = "none";
            e.preventDefault();
            e.stopPropagation();
        });

        document.addEventListener("pointermove", (e) => {
            if (this.resizing) {
                const rect = this.container.getBoundingClientRect();
                const newHeight = e.clientY - rect.top;
                this.container.style.height = `${newHeight}px`;
                e.preventDefault();
                e.stopPropagation();
            }
        });

        document.addEventListener("pointerup", () => {
            this.resizing = false;
            if (this.img && !this.no_scribbles) scribbleIndicator.style.display = "inline-block";
        });

        toolbar.addEventListener("pointerdown", (e) => {
            e.stopPropagation();
        });

        this.container.addEventListener("wheel", (e) => {
            if (!this.img) return;
            e.preventDefault();
            const delta = e.deltaY * -0.001;
            let scale = true;

            if (this._held_W) { // Width
                scribbleWidth.value = parseInt(scribbleWidth.value) - Math.sign(e.deltaY);
                updateInput(scribbleWidth);
                scale = false;
            }
            if (this._held_A) { // Alpha (Opacity)
                scribbleAlpha.value = parseInt(scribbleAlpha.value) - Math.sign(e.deltaY) * 5;
                updateInput(scribbleAlpha);
                scale = false;
            }
            if (this._held_S) { // Softness
                scribbleSoftness.value = parseInt(scribbleSoftness.value) - Math.sign(e.deltaY) * 5;
                updateInput(scribbleSoftness);
                scale = false;
            }

            if (!scale) return;

            const rect = this.container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const oldScale = this.imgScale;
            this.imgScale += delta;
            this.imgScale = Math.max(0.1, this.imgScale);
            const newScale = this.imgScale / oldScale;
            this.imgX = x - (x - this.imgX) * newScale;
            this.imgY = y - (y - this.imgY) * newScale;
            this.drawImage();
        });

        this.container.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            this.dragged_just_now = false;
            return false;
        });

        this.container.addEventListener("pointerover", () => {
            toolbar.style.opacity = "1";
            if (!this.img && !this.no_upload) this.container.style.cursor = "pointer";
            if (this.img && !this.no_scribbles) scribbleIndicator.style.display = "inline-block";
        });


        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        for (const e of ["dragenter", "dragover", "dragleave", "drop"]) {
            this.container.addEventListener(e, preventDefaults, false);
        }

        this.container.addEventListener("dragleave", () => {
            toolbar.style.opacity = "0";
            imageContainer.style.cursor = "";
            drawingCanvas.style.cursor = "";
            this.container.style.cursor = "";
            scribbleIndicator.style.display = "none";
        });

        this.container.addEventListener("dragenter", () => {
            imageContainer.style.cursor = "copy";
            drawingCanvas.style.cursor = "copy";
        });

        this.container.addEventListener("drop", (e) => {
            imageContainer.style.cursor = "pointer";
            drawingCanvas.style.cursor = "crosshair";
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
                if (this.img && !this.no_scribbles) scribbleIndicator.style.display = "inline-block";
            }
        });

        this.container.addEventListener("pointerenter", () => {
            this.pointerInsideContainer = true;
            if (this.img && !this.no_scribbles) scribbleIndicator.style.display = "inline-block";
        });

        this.container.addEventListener("pointerleave", () => {
            this.pointerInsideContainer = false;
            scribbleIndicator.style.display = "none";
        });

        document.addEventListener("paste", (e) => { // event listener on container instead?
            if (this.pointerInsideContainer) {
                this.handlePaste(e);
                if (this.img && !this.no_scribbles) scribbleIndicator.style.display = "inline-block";
            }
        });

        document.addEventListener("keydown", (e) => {
            if (!this.pointerInsideContainer) return;
            if (e.ctrlKey && e.key === "z") {
                e.preventDefault();
                this.undo();
            }
            else if (e.ctrlKey && e.key === "y") {
                e.preventDefault();
                this.redo();
            }

            if (e.key === "w" && !this.scribbleWidthFixed)    this._held_W = true;
            if (e.key === "a" && !this.scribbleAlphaFixed)    this._held_A = true;
            if (e.key === "s" && !this.scribbleSoftnessFixed) this._held_S = true;
        });

        document.addEventListener("keyup", (e) => {
            this._held_W = false;
            this._held_A = false;
            this._held_S = false;

            if (!this.pointerInsideContainer) return;

            if (e.ctrlKey && e.key === "x") {
                e.preventDefault();
                this.resetImage();
            }
// Ctrl+Q to remove ? maybe not useful
// Ctrl+O to open ?

            if (e.key === "r") centerButton.click();

            if (e.key === "f") this.toggleMaximize();
            
            if (e.key === "e" && !this.scribbleColorFixed) scribbleColor.click();
        });

        this.maxButton.addEventListener("click", () => {
            this.toggleMaximize();
        });

        this.eraserButton.addEventListener("click", () => {
            this.toggleEraser();
        });

        this.updateUndoRedoButtons();

        this.background_gradio_bind.listen((value) => {
            this.loadImage(value);
        });

        this.foreground_gradio_bind.listen((value) => {
            this.loadDrawing(value);
        });
    }

    handleDraw(e) {
        const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const ctx = canvas.getContext("2d");
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.imgScale;
        const y = (e.clientY - rect.top) / this.imgScale;

        this.temp_draw_points.push([x, y]);
        ctx.putImageData(this.temp_draw_bg, 0, 0);
        ctx.beginPath();
        ctx.moveTo(this.temp_draw_points[0][0], this.temp_draw_points[0][1]);

        for (let i = 1; i < this.temp_draw_points.length; i++) {
            ctx.lineTo(this.temp_draw_points[i][0], this.temp_draw_points[i][1]);
        }

        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = Math.max(1, (this.scribbleWidth / this.imgScale) * 20);

        if (this.erasing) {
            ctx.globalCompositeOperation = "destination-out";
        }
        else {
            ctx.globalCompositeOperation = "source-over";
        }

        if (this.contrast_scribbles) {
            ctx.strokeStyle = this.contrast_pattern;
            ctx.stroke();
            return;
        }

        ctx.strokeStyle = this.scribbleColor;

        if (this.scribbleSoftness <= 0) {
            ctx.globalAlpha = this.scribbleAlpha / 100;
            ctx.stroke();
            return;
        }

        const innerWidth = ctx.lineWidth * (1 - this.scribbleSoftness / 96);
        const outerWidth = ctx.lineWidth * (1 + this.scribbleSoftness / 96);
        const steps = Math.round(5 + this.scribbleSoftness / 5);
        const stepWidth = (outerWidth - innerWidth) / (steps - 1);

        ctx.globalAlpha = 1 - Math.pow(1 - Math.min(this.scribbleAlpha / 100, 0.95), 1 / steps);

        for (let i = 0; i < steps; i++) {
            ctx.lineWidth = innerWidth + stepWidth * i;
            ctx.stroke();
        }
    }

    handleFileUpload(file) {
        if (file && !this.no_upload) {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.loadImage(e.target.result);
            };
            reader.readAsDataURL(file);
        }
    }

    handlePaste(e) {
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.type.indexOf("image") !== -1) {
                const file = item.getAsFile();
                this.handleFileUpload(file);
                break;
            }
        }
    }

    loadImage(base64) {
        if (typeof this.gradio_config !== "undefined") {
            if (!this.gradio_config.version.startsWith("4.")) return;
        } else {
            return;
        }

        const image = new Image();
        image.onload = () => {
            this.img = base64;
            this.orgWidth = image.width;
            this.orgHeight = image.height;
            const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            canvas.width = image.width;
            canvas.height = image.height;
            this.adjustInitialPositionAndScale();
            this.drawImage();
            this.updateBackgroundImageData();
            this.history = [];
            this.historyIndex == -1;
            this.saveState();
            document.getElementById(`imageInput_${this.uuid}`).value = null;
            document.getElementById(`uploadHint_${this.uuid}`).style.display = "none";
        };

        if (base64) {
            image.src = base64;
        } else {
            this.img = null;
            const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            canvas.width = 1;
            canvas.height = 1;
            this.adjustInitialPositionAndScale();
            this.drawImage();
            this.saveState();
        }
    }

    loadDrawing(base64) {
        const image = new Image();
        image.onload = () => {
            const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(image, 0, 0);
            this.history = [];
            this.historyIndex == -1;
            this.saveState();
        };
        if (base64) {
            image.src = base64;
        } else {
            const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            this.saveState();
        }
    }

    isInsideImage(x, y) {
        const scaledWidth = this.orgWidth * this.imgScale;
        const scaledHeight = this.orgHeight * this.imgScale;
        return x > this.imgX && x < this.imgX + scaledWidth && y > this.imgY && y < this.imgY + scaledHeight;
    }

    drawImage() {
        const image = document.getElementById(`image_${this.uuid}`);
        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        if (this.img) {
            const scaledWidth = this.orgWidth * this.imgScale;
            const scaledHeight = this.orgHeight * this.imgScale;
            image.src = this.img;
            image.style.width = `${scaledWidth}px`;
            image.style.height = `${scaledHeight}px`;
            image.style.left = `${this.imgX}px`;
            image.style.top = `${this.imgY}px`;
            image.style.display = "block";
            drawingCanvas.style.width = `${scaledWidth}px`;
            drawingCanvas.style.height = `${scaledHeight}px`;
            drawingCanvas.style.left = `${this.imgX}px`;
            drawingCanvas.style.top = `${this.imgY}px`;
        } else {
            image.src = "";
            image.style.display = "none";
        }
    }

    adjustInitialPositionAndScale() {
        const containerWidth = this.container.clientWidth - 20;
        const containerHeight = this.container.clientHeight - 20;
        const scaleX = containerWidth / this.orgWidth;
        const scaleY = containerHeight / this.orgHeight;
        this.imgScale = Math.min(scaleX, scaleY);
        const scaledWidth = this.orgWidth * this.imgScale;
        const scaledHeight = this.orgHeight * this.imgScale;
        this.imgX = (this.container.clientWidth - scaledWidth) / 2;
        this.imgY = (this.container.clientHeight - scaledHeight) / 2;
    }

    resetImage() {
        if (this.img) {
            const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            this.adjustInitialPositionAndScale();
            this.drawImage();
            this.saveState();
        }
    }

    removeImage() {
        this.img = null;
        const image = document.getElementById(`image_${this.uuid}`);
        const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        image.src = "";
        image.style.width = "0";
        image.style.height = "0";
        this.history = [];
        this.historyIndex = -1;
        this.saveState();
        if (!this.no_upload) {
            document.getElementById(`uploadHint_${this.uuid}`).style.display = "inline-block";
        }
        this.loadImage(null);
        this.updateUndoRedoButtons();
    }

    saveState() {
        const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const ctx = canvas.getContext("2d");
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        this.history = this.history.slice(0, this.historyIndex + 1);
        this.history.push(imageData);
        this.historyIndex++;
        this.updateUndoRedoButtons();
        this.updateDrawingData();
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.restoreState();
            this.updateUndoRedoButtons();
        }
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.restoreState();
            this.updateUndoRedoButtons();
        }
    }

    restoreState() {
        const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const ctx = canvas.getContext("2d");
        const imageData = this.history[this.historyIndex];
        ctx.putImageData(imageData, 0, 0);
        this.updateDrawingData();
    }

    updateUndoRedoButtons() {
        const undoButton = document.getElementById(`undoButton_${this.uuid}`);
        const redoButton = document.getElementById(`redoButton_${this.uuid}`);
        undoButton.disabled = this.historyIndex <= 0;
        redoButton.disabled = this.historyIndex >= this.history.length - 1;
        undoButton.style.opacity = undoButton.disabled ? "0.5" : "1";
        redoButton.style.opacity = redoButton.disabled ? "0.5" : "1";
    }

    updateBackgroundImageData() {
        if (!this.img) {
            this.background_gradio_bind.set_value("");
            return;
        }
        const image = document.getElementById(`image_${this.uuid}`);
        const tempCanvas = this.temp_canvas;
        const tempCtx = tempCanvas.getContext("2d");
        tempCanvas.width = this.orgWidth;
        tempCanvas.height = this.orgHeight;
        tempCtx.drawImage(image, 0, 0, this.orgWidth, this.orgHeight);
        const dataUrl = tempCanvas.toDataURL("image/png");
        this.background_gradio_bind.set_value(dataUrl);
    }

    updateDrawingData() {
        if (!this.img) {
            this.foreground_gradio_bind.set_value("");
            return;
        }
        const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const dataUrl = canvas.toDataURL("image/png");
        this.foreground_gradio_bind.set_value(dataUrl);
    }

    toggleMaximize() {
        if (this.maximized) {
            this.container.style.width = this.originalState.width;
            this.container.style.height = this.originalState.height;
            this.container.style.top = this.originalState.top;
            this.container.style.left = this.originalState.left;
            this.container.style.position = this.originalState.position;
            this.container.style.zIndex = this.originalState.zIndex;
            this.maxButton.innerText = "⛶";
            this.maximized = false;
        }
        else {
            this.originalState = {
                width:    this.container.style.width,
                height:   this.container.style.height,
                top:      this.container.style.top,
                left:     this.container.style.left,
                position: this.container.style.position,
                zIndex:   this.container.style.zIndex,
            };

            this.container.style.width = "100vw";
            this.container.style.height = "100vh";
            this.container.style.top = "0";
            this.container.style.left = "0";
            this.container.style.position = "fixed";
            this.container.style.zIndex = "1000";
            this.maxButton.innerText = "➖";
            this.maximized = true;
        }
    }
    
    toggleEraser() {
        if (this.erasing) {
            this.erasing = false;
            this.eraserButton.style.transform = "";
            this.eraserButton.style.outline = "0px";
        } else {
            this.erasing = true;
            this.eraserButton.style.transform = "rotate(180deg)";
            this.eraserButton.style.outline = "1px solid red";
        }
    }
}

const True = true;
const False = false;
