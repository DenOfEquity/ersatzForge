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
