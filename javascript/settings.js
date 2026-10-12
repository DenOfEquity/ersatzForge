let settingsExcludeTabsFromShowAll = {
    settings_tab_defaults: 1,
    settings_tab_sysinfo: 1,
    settings_tab_actions: 1,
    settings_tab_licenses: 1,
};

function settingsShowAllTabs() {
    gradioApp().querySelectorAll("#settings > div").forEach(function(elem) {
        if (settingsExcludeTabsFromShowAll[elem.id]) return;

        elem.classList.add("settings_tab_visible");
    });
}

function settingsShowOneTab() {
    gradioApp().querySelectorAll("#settings > div").forEach(function(elem) {
        if (!elem.selected) {
            elem.classList.remove("settings_tab_visible");
        }
    });
}

onUiLoaded(function() {
    var editTextarea = gradioApp().querySelector("#settings_search > label > input");

    onEdit("settingsSearch", editTextarea, 250, function() {
        var searchText = (editTextarea.value || "").trim().toLowerCase();
        
        if (searchText.length >= 3) {
            gradioApp().querySelectorAll("#settings > div[id^=settings_] div[id^=column_settings_] > *").forEach(function(elem) {
                var visible = elem.textContent.trim().toLowerCase().indexOf(searchText) != -1;
                elem.style.display = visible ? "" : "none";
            });
            settingsShowAllTabs();
        }
        else {
            gradioApp().querySelectorAll("#settings > div[id^=settings_] div[id^=column_settings_] > *").forEach(function(elem) {
                elem.style.display = "";
            });
            settingsShowOneTab();
        }
    });
});


onOptionsChanged(function() {
    if (gradioApp().querySelector('#settings .settings-category')) return;

    var sectionMap = {};
    gradioApp().querySelectorAll('#settings > div > button').forEach(function(x) {
        sectionMap[x.textContent.trim()] = x;
    });

    opts._categories.forEach(function(x) {
        var section = localization[x[0]] ?? x[0];
        var category = localization[x[1]] ?? x[1];

        var span = document.createElement('SPAN');
        span.textContent = category;
        span.className = 'settings-category';

        var sectionElem = sectionMap[section];
        if (!sectionElem) return;

        sectionElem.parentElement.insertBefore(span, sectionElem);
    });
});

