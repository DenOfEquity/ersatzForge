// code related to showing and updating progressbar shown as the image is being made

function rememberGallerySelection() {

}

function getGallerySelectedIndex() {

}

function request(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function pad2(x) {
    return x < 10 ? '0' + x : x;
}

function formatTime(secs) {
    if (secs > 3600) {
        return pad2(Math.floor(secs / 60 / 60)) + ":" + pad2(Math.floor(secs / 60) % 60) + ":" + pad2(Math.floor(secs) % 60);
    } else if (secs > 60) {
        return pad2(Math.floor(secs / 60)) + ":" + pad2(Math.floor(secs) % 60);
    } else {
        return Math.floor(secs) + "s";
    }
}


var originalAppTitle = undefined;

onUiLoaded(function() {
    originalAppTitle = document.title;
});

function setTitle(progress) {
    var title = originalAppTitle;

    if (opts.show_progress_in_title && progress) {
        title = '[' + progress.trim() + '] ' + title;
    }

    if (document.title != title) {
        document.title = title;
    }
}


function randomId() {
    return "task(" + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + ")";
}

var livePreview = document.createElement('div');
livePreview.className = 'livePreview';
livePreview.style.cursor = 'pointer';
livePreview.style.userSelect = 'none';
livePreview.addEventListener('click', function(evt) {
    if (!opts.js_modal_lightbox || evt.button != 0) return;

    modalZoomSet(opts.js_modal_lightbox_initially_zoomed);
    evt.preventDefault();
    showModal(evt);
}, true);
livePreview.appendChild(document.createElement('p'));	// dummy element, will be replaced by img


// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and
// preview inside gallery element. Cleans up all created stuff when the task is over and calls atEnd.
// calls onProgress every time there is a progress update
function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 40) {
    var dateStart = new Date();
    var wasEverActive = false;
    var parentProgressbar = progressbarContainer.parentNode;
    var wakeLock = null;

    var requestWakeLock = async function() {
        if (!opts.prevent_screen_sleep_during_generation || wakeLock !== null) return;
        try {
            wakeLock = await navigator.wakeLock.request('screen');
        } catch (err) {
            console.error('Wake Lock is not supported.');
            wakeLock = false;
        }
    };

    var releaseWakeLock = async function() {
        if (!opts.prevent_screen_sleep_during_generation || !wakeLock) return;
        try {
            await wakeLock.release();
            wakeLock = null;
        } catch (err) {
            console.error('Wake Lock release failed', err);
        }
    };

    var divProgress = document.createElement('div');
    divProgress.className = 'progressDiv';
    divProgress.style.display = opts.show_progressbar ? "block" : "none";
    var divInner = document.createElement('div');
    divInner.className = 'progress';

    divProgress.appendChild(divInner);
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    var removeProgressBar = function() {
        releaseWakeLock();
        if (!divProgress) return;

        setTitle("");
        if (parentProgressbar.contains(divProgress))     parentProgressbar.removeChild(divProgress);
        if (gallery && gallery.contains(livePreview))    gallery.removeChild(livePreview);

        atEnd();

        divProgress = null;
    };

    var funProgress = function(id_task) {
        requestWakeLock();
        request("./internal/progress", {id_task: id_task, live_preview: false}, function(res) {
            if (res.completed) {
                removeProgressBar();
                return;
            }

            let progressText = "";

            divInner.style.width = ((res.progress || 0) * 100.0) + '%';
            divInner.style.background = res.progress ? "" : "transparent";

            if (res.progress > 0) {
                progressText = ((res.progress || 0) * 100.0).toFixed(0) + '%';
            }

            if (res.eta) {
                progressText += " ETA: " + formatTime(res.eta);
            }

            setTitle(progressText);

            if (res.textinfo && res.textinfo.indexOf("\n") == -1) {
                progressText = res.textinfo + " " + progressText;
            }

            divInner.textContent = progressText;

            var elapsedFromStart = (new Date() - dateStart) / 1000;

            if (res.active) wasEverActive = true;

            if (!res.active && wasEverActive) {
                removeProgressBar();
                return;
            }

            if (elapsedFromStart > inactivityTimeout && !res.queued && !res.active) {
                removeProgressBar();
                return;
            }

            if (onProgress) {
                onProgress(res);
            }

            setTimeout(() => {
                funProgress(id_task);
            }, opts.live_preview_refresh_period || 500);
        }, function() {
            removeProgressBar();
        });
    };

    var funLivePreview = function(id_task, id_live_preview) {
        request("./internal/progress", {id_task: id_task, id_live_preview: id_live_preview}, function(res) {
            if (!divProgress) {
                return;
            }

            if (res.live_preview && gallery) {
                var img = new Image();
                img.onload = function() {
                    livePreview.replaceChild(img, livePreview.lastElementChild)
                    gallery.insertBefore(livePreview, gallery.firstElementChild);
                };
                img.src = res.live_preview;
				if (opts.js_live_preview_in_modal_lightbox) {
				    lightBoxImage.src = img.src;
					if (lightBoxImage.style.display === 'none') {
						lightBoxModal.style.setProperty('background-image', `url(${lightBoxImage.src})`);
					}
                }
				else {
					let currentButton = selected_gallery_button();
					if (currentButton?.children?.length > 0) {
						lightBoxImage.src = currentButton.children[0].src;
					}
					if (lightBoxImage.style.display === 'none') {
						lightBoxModal.style.setProperty('background-image', `url(${lightBoxImage.src})`);
					}
				}
            }

            setTimeout(() => {
                funLivePreview(id_task, res.id_live_preview);
            }, opts.live_preview_refresh_period || 500);
        }, function() {
            removeProgressBar();
        });
    };

    funProgress(id_task, 0);

    if (gallery) {
        funLivePreview(id_task, 0);
    }

}
