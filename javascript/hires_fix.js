
function onCalcResolutionHires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y) {
    function setInactive(elem, inactive) {
        elem.classList.toggle("inactive", !!inactive);
    }

    var hrUpscaleBy = gradioApp().getElementById("txt2img_hr_scale");
    var hrResizeX = gradioApp().getElementById("txt2img_hr_resize_x");
    var hrResizeY = gradioApp().getElementById("txt2img_hr_resize_y");

    setInactive(hrUpscaleBy, hr_resize_x > 0 || hr_resize_y > 0);
    setInactive(hrResizeX, hr_resize_x == 0);
    setInactive(hrResizeY, hr_resize_y == 0);

    return [enable, width, height, hr_scale, hr_resize_x, hr_resize_y];
}
