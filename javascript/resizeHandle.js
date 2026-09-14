(function() {
    const GRADIO_MIN_WIDTH = 320;
    const PAD = 16;
    const DEBOUNCE_TIME = 100;

    const R = {
        tracking: false,
        parent: null,
        parentWidth: null,
        leftColStartWidth: null,
        screenX: null,
    };

    let resizeTimer;
    let parents = [];


    function displayResizeHandle(parent) {
        if (!parent.needHideOnMobile) {
            return true;
        }
        if (window.innerWidth < GRADIO_MIN_WIDTH * 2 + PAD * 4) {
            parent.style.display = "flex";
            parent.resizeHandle.style.display = "none";
            return false;
        }
        else {
            parent.style.display = "grid";
            parent.resizeHandle.style.display = "block";
            return true;
        }
    }

    function afterResize(parent) {
        if (displayResizeHandle(parent) && parent.style.gridTemplateColumns != parent.style.originalGridTemplateColumns) {
            const oldParentWidth = R.parentWidth;
            const newParentWidth = parent.offsetWidth;
            const widthL = parseInt(parent.style.gridTemplateColumns.split(" ")[0]);

            const ratio = newParentWidth / oldParentWidth;

            const newWidthL = Math.max(Math.floor(ratio * widthL), parent.minLeftColWidth);
            parent.style.gridTemplateColumns = `${newWidthL}px 16px 1fr`;

            R.parentWidth = newParentWidth;
        }
    }

    function setup(parent) {
        const leftCol = parent.firstElementChild;
        const rightCol = parent.lastElementChild;

        parents.push(parent);

        parent.style.display = "grid";
        parent.style.gap = "0";
        let leftColTemplate = "";
        if (parent.children[0].style.flexGrow) {
            leftColTemplate = `${parent.children[0].style.flexGrow}fr`;
            parent.minLeftColWidth = GRADIO_MIN_WIDTH;
            parent.minRightColWidth = GRADIO_MIN_WIDTH;
            parent.needHideOnMobile = true;
        }
        else {
            leftColTemplate = parent.children[0].style.flexBasis;
            parent.minLeftColWidth = parent.children[0].style.flexBasis.slice(0, -2) / 2;
            parent.minRightColWidth = 0;
            parent.needHideOnMobile = false;
        }

        if (!leftColTemplate) {
            leftColTemplate = "1fr";
        }

        const gridTemplateColumns = `${leftColTemplate} ${PAD}px ${parent.children[1].style.flexGrow}fr`;
        parent.style.gridTemplateColumns = gridTemplateColumns;
        parent.style.originalGridTemplateColumns = gridTemplateColumns;

        const resizeHandle = document.createElement("div");
        resizeHandle.classList.add("resize-handle");
        parent.insertBefore(resizeHandle, rightCol);
        parent.resizeHandle = resizeHandle;


        function startTracking(event, X) {
            event.preventDefault();
            event.stopPropagation();
            document.body.classList.add("resizing");

            R.tracking = true;
            R.parent = parent;
            R.parentWidth = parent.offsetWidth;
            R.leftColStartWidth = leftCol.offsetWidth;
            R.screenX = X;
        }

        resizeHandle.addEventListener("mousedown", (evt) => {
            if (evt.button === 0) startTracking(evt, evt.screenX);
        }, {passive: false});

        resizeHandle.addEventListener("touchstart", (evt) => {
            if (evt.changedTouches.length === 1) startTracking(evt, evt.changedTouches[0].screenX);
        }, {passive: false});

        resizeHandle.addEventListener("dblclick", (evt) => {
            evt.preventDefault();
            evt.stopPropagation();

            parent.style.gridTemplateColumns = parent.style.originalGridTemplateColumns;
        });

        afterResize(parent);
    }


    function doTracking(event, X) {
        if (R.tracking) {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();

            let delta = R.screenX - X;
            const leftColWidth = Math.max(Math.min(R.leftColStartWidth - delta, R.parent.offsetWidth - R.parent.minRightColWidth - PAD), R.parent.minLeftColWidth);
            R.parent.style.gridTemplateColumns = `${leftColWidth}px 16px 1fr`;
        }
    }

    window.addEventListener("mousemove", (evt) => {
        if (evt.button === 0) doTracking(evt, evt.screenX);
    });

    window.addEventListener("touchmove", (evt) => {
        if (evt.changedTouches.length === 1) doTracking(evt, evt.changedTouches[0].screenX);
    });


    function stopTracking(event) {
        if (R.tracking) {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            document.body.classList.remove("resizing");

            R.tracking = false;
        }
    }

    window.addEventListener("mouseup", (evt) => {
        if (evt.button === 0) stopTracking(evt);
    });
    window.addEventListener("touchend", (evt) => {
        if (evt.changedTouches.length === 1) stopTracking(evt);
    });


    window.addEventListener("resize", () => {
        clearTimeout(resizeTimer);

        resizeTimer = setTimeout(function() {
            for (const parent of parents) {
                afterResize(parent);
            }
        }, DEBOUNCE_TIME);
    });

    setupResizeHandle = setup;
})();


function setupAllResizeHandles() {
    for (var elem of gradioApp().querySelectorAll('.resize-handle-row')) {
        if (!elem.querySelector('.resize-handle') && !elem.children[0].classList.contains("hidden")) {
            setupResizeHandle(elem);
        }
    }
}


onUiLoaded(setupAllResizeHandles);
