# Noise Colouriser #
### extension for Forge webui for Stable Diffusion ###
---

ways of manipulating the starting noise, mainly colourising it. There are extensions which do this already ([example](https://github.com/kenning/sd-webui-noise-color-picker)), but they switch to img2img processing behind the scenes, which is excellent cheating. 

Some samplers reduce/block the effect: DPM++ SDE, UniPC. Effect, especially ideal strength, also highly dependent on model.

Make your own custom presets by renaming/copying `colourPresets.py`, in the extension directory, to `customPresets.py`. Changes to the original will be overwritten by updates, but the custom version will be used preferentially.


**⨁** button toggles centre to mean of initial noise.

**∿** button toggles low-discrepancy noise

**♯** button toggles sharpening of initial noise.

**🎯** button toggles a different noise adjustment method.

**E** button toggles applying colour every step. If enabled, start step setting is ignored.
