## First Block Cache and TeaCache, in ersatzForge webUI ##
### accelerate inference at some, perhaps minimal, quality cost ###

this version compatible with ersatzForge only

derived, with lots of reworking, from:
* https://github.com/likelovewant/sd-forge-teacache (flux only, teaache only)

more info:
* https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4FLUX
* https://github.com/chengzeyi/Comfy-WaveSpeed


usage:
1. Enable the extension
2. select caching threshold: higher threshold = more caching = faster + lower quality
3. low step models (Hyper/Turbo/Lightning/Schnell) will need higher threshold to do anything
4. Generate
5 You'll need to experiment to find settings that work with your favoured models, step counts, samplers.

>[!NOTE]
>Both methods work with SD1.5, SD2, SDXL (including separated cond processing), and Flux.
>also added versions for SD3 and Chroma. Caching SD3 does not seem to work especially well, but may be more useful with higher steps.
>The use of cached residuals applies to the whole batch, so results will not be identical between different batch sizes. This is absolutely 100% *will not fix*.

---
