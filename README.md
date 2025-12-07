a backup of my local (experimental, opinionated) changes to Forge2 webUI
* broadly similiar package requirements to original Forge - no new Python, Numpy, etc. I'm using Torch 2.6 and CUDA 12.4 on a GTX1070.
* auto selection of VAE and text encoders per model / UI setting
* Chroma (based on https://github.com/croquelois/forgeChroma) and Chroma Radiance (based on https://github.com/maybleMyers/chromaforge)
* extended Checkpoint Merger, including UI for nArn0's embedding convertor (based on https://github.com/nArn0/sdxl-embedding-converter)
* Hypernetworks
* various other tweaks: UI, embedding filtering, code consolidation and tidying, dead code removal, performance improvements (for me)
* tiling (sd1, 2, xl) (based on https://github.com/spinagon/ComfyUI-seamless-tiling)
* all embeddings everywhere all at once: SD1.5 embeddings (CLIP-L only) can be used with SDXL, SD3; SDXL embeddings can be used with SD1 (applies CLIP-L only, CLIP-G ignored), SD3
* new preprocessors for IPAdapter, including tiling, noising (for uncond) and sharpening of inputs. And multi-input.
* Latent NeuralNet upscaler by city96 (based on https://github.com/city96/SD-Latent-Upscaler) (1.25x, 1.5x and 2.0x)
* support more upscaler models via updated Spandrel - place models in `models/upscaler`.
* ResAdapter support: download models to `models/other_modules`, load via 'Additional modules' selector (as VAE, text encoder), LoRA as usual (https://huggingface.co/jiaxiangc/res-adapter)
* long CLIP
* distilled T5 models for Flux by LifuWang (see https://huggingface.co/LifuWang/DistillT5)
* lama and MAT inpainting models usable in img2img, both as processing options and as infill options
* PuLID (sdxl) (based on https://github.com/cubiq/PuLID_ComfyUI/)
* nVidia Cosmos predict2 t2i (only tested 2B model)
* Wan 2.2 14B (based on Haoming02's implementation in [ForgeNeo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)) and 5B, 1 frame generation only
* ELLA (https://github.com/TencentQQGYLab/ELLA), models downloaded automatically for first use, enable in Settings
* extra samplers: SA-Solver, ER-SDE, Adaptive-ODE (params in Settings), Fixed-ODE; and extra options/variants.
* assorted extra functionality in built-in extensions, controlnet preprocessors
* built-in latent manipulations such as Epsilon Scaling, CFG rescale, CFG normalization (see Settings)
* dyPE high resolution generation for Flux (based on https://github.com/guyyariv/DyPE), works as far as I can test on my limited hardware
* Lumina2
* Z-Image-Turbo (1024x1024, fp8 model - 15s per iteration ⇒ 2:15 for 9 step generation on old laptop with 8GB VRAM GTX1070); including Union Control.
