# coding=utf-8
# Qwen3-TTS Gradio Demo for HuggingFace Spaces with Zero GPU
# Supports: Voice Design, Voice Clone (Base), TTS (CustomVoice)

import os
import gc
import gradio as gr
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from huggingface_hub import snapshot_download


# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


voice_design_model = None
base_model_1_7b = None
custom_voice_model_1_7b = None


def unload():
    global voice_design_model, custom_voice_model_1_7b, base_model_1_7b
    voice_design_model = None
    base_model_1_7b = None
    custom_voice_model_1_7b = None
    torch.cuda.empty_cache()
    gc.collect()


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def generate_voice_design(text, language, voice_description):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    global voice_design_model
    if voice_design_model is None:
        unload()
        voice_design_model = Qwen3TTSModel.from_pretrained(
            snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
            dtype=torch.bfloat16,     device_map="cuda",
            # dtype=torch.float32,     device_map="cuda",
            attn_implementation="eager"
        )

    wavs, sr = voice_design_model.generate_voice_design(
        text=text.strip(),
        language=language,
        instruct=voice_description.strip(),
        non_streaming_mode=True,
        max_new_tokens=2048,
    )
    return (sr, wavs[0]), "Voice design generation completed successfully!", gr.Button.update(interactive=True), gr.Button.update(interactive=True), gr.Button.update(interactive=True)


cloned_voice = {}

def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, save_cloned="", use_pre_clone=""):
    """Generate speech using Base (Voice Clone) model."""
    global cloned_voice

    save_cloned = save_cloned.strip()

    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    if use_pre_clone != "" and use_pre_clone in cloned_voice:
        audio_tuple = None
        ref_text = None
    else:
        audio_tuple = _audio_to_tuple(ref_audio)
        if audio_tuple is None:
            return None, "Error: Reference audio is required."

        if not use_xvector_only and (not ref_text or not ref_text.strip()):
            return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."

    global base_model_1_7b
    if base_model_1_7b is None:
        unload()
        base_model_1_7b = Qwen3TTSModel.from_pretrained(
            snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            dtype=torch.bfloat16,     device_map="cuda",
        )

    # wavs, sr = base_model_1_7b.generate_voice_clone(
    wavs, sr, cloned = base_model_1_7b.generate_voice_clone(
        text=target_text.strip(),
        language=language,
        ref_audio=audio_tuple,
        ref_text=ref_text.strip() if ref_text else None,
        x_vector_only_mode=use_xvector_only,
        max_new_tokens=2048,
        voice_clone_prompt=cloned_voice.get(use_pre_clone), 
    )

    if save_cloned != "" and save_cloned != use_pre_clone:
        cloned_voice[save_cloned] = cloned

# add another tab to merging cloned voices?

    return (sr, wavs[0]), "Voice clone generation completed successfully!", gr.Button.update(interactive=True), gr.Button.update(interactive=True), gr.Button.update(interactive=True)


def generate_custom_voice(text, language, speaker, instruct):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."

    global custom_voice_model_1_7b
    if custom_voice_model_1_7b is None:
        unload()
        custom_voice_model_1_7b = Qwen3TTSModel.from_pretrained(
            snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            dtype=torch.bfloat16,     device_map="cuda",
        )

    wavs, sr = custom_voice_model_1_7b.generate_custom_voice(
        text=text.strip(),
        language=language,
        speaker=speaker.lower().replace(" ", "_"),
        instruct=instruct.strip() if instruct else None,
        non_streaming_mode=True,
        max_new_tokens=2048,
    )
    return (sr, wavs[0]), "Generation completed successfully!", gr.Button.update(interactive=True), gr.Button.update(interactive=True), gr.Button.update(interactive=True)


# Build Gradio UI
theme = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
)

css = """
.gradio-container {max-width: none !important;}
.tab-content {padding: 20px;}
footer {display: none !important;}
"""

with gr.Blocks(analytics_enabled=False, theme=theme, css=css, title="Qwen3-TTS Demo") as demo:
    gr.Markdown("# Qwen3-TTS 1.7B - Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team.")

    def status_working():
        return "Processing ...", gr.Button.update(interactive=False), gr.Button.update(interactive=False), gr.Button.update(interactive=False)

    with gr.Tabs():
        # Tab 1: Voice Design (Default, 1.7B only)
        with gr.Tab("Voice Design"):
            gr.Markdown("### Create Custom Voice with Natural Language")
            with gr.Row():
                with gr.Column(scale=2):
                    design_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter the text you want to convert to speech...",
                        value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
                    )
                    design_language = gr.Dropdown(
                        label="Language",
                        choices=LANGUAGES,
                        value="Auto",
                        interactive=True,
                    )
                    design_instruct = gr.Textbox(
                        label="Voice Description",
                        lines=3,
                        placeholder="Describe the voice characteristics you want...",
                        value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
                    )

                with gr.Column(scale=2):
                    design_btn = gr.Button("Generate with Custom Voice", variant="primary")
                    design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    design_status = gr.Textbox(label="Status", lines=2, interactive=False)


        # Tab 2: Voice Clone (Base)
        with gr.Tab("Voice Clone (Base)"):
            gr.Markdown("### Clone Voice from Reference Audio")
            with gr.Row():
                with gr.Column(scale=2):
                    clone_ref_audio = gr.Audio(
                        label="Reference Audio (Upload a voice sample to clone)",
                        type="numpy",
                    )
                    clone_ref_text = gr.Textbox(
                        label="Reference Text (Transcript of the reference audio)",
                        lines=2,
                        placeholder="Enter the exact text spoken in the reference audio...",
                    )
                    clone_xvector = gr.Checkbox(
                        label="Use x-vector only (No reference text needed, but lower quality)",
                        value=False,
                    )

                    with gr.Row():
                        save_cloned = gr.Textbox(label="Store cloned voice as {name}", max_lines=1, value="")
                        use_cloned = gr.Textbox(label="Reuse cloned voice {name}", max_lines=1, value="")
                        show_cloned = gr.Button(value="List clones", scale=0)


                    clone_target_text = gr.Textbox(
                        label="Target Text (Text to synthesize with cloned voice)",
                        lines=4,
                        placeholder="Enter the text you want the cloned voice to speak...",
                    )
                    with gr.Row():
                        clone_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="Auto",
                            interactive=True,
                        )

                with gr.Column(scale=2):
                    clone_btn = gr.Button("Clone & Generate", variant="primary")
                    clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    clone_status = gr.Textbox(label="Status", lines=2, interactive=False)

            def list_clones():
                global cloned_voice
                print ("Qwen3-TTS cloned voice list:")
                for cv in cloned_voice.keys():
                    print (f"    {cv}")
                return "Named clones: " + ", ".join(cloned_voice.keys())

            show_cloned.click(fn=list_clones, inputs=None, outputs=clone_status)

        # Tab 3: TTS (CustomVoice)
        with gr.Tab("TTS (CustomVoice)"):
            gr.Markdown("### Text-to-Speech with Predefined Speakers")
            with gr.Row():
                with gr.Column(scale=2):
                    tts_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter the text you want to convert to speech...",
                        value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities."
                    )
                    with gr.Row():
                        tts_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="English",
                            interactive=True,
                        )
                        tts_speaker = gr.Dropdown(
                            label="Speaker",
                            choices=SPEAKERS,
                            value="Ryan",
                            interactive=True,
                        )
                    with gr.Row():
                        tts_instruct = gr.Textbox(
                            label="Style Instruction (Optional)",
                            lines=2,
                            placeholder="e.g., Speak in a cheerful and energetic tone",
                        )

                with gr.Column(scale=2):
                    tts_btn = gr.Button("Generate Speech", variant="primary")
                    tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

        design_btn.click(status_working, inputs=None, outputs=[design_status, design_btn, clone_btn, tts_btn], show_progress="hidden").then(
            generate_voice_design,
            inputs=[design_text, design_language, design_instruct],
            outputs=[design_audio_out, design_status, design_btn, clone_btn, tts_btn],
            show_progress="minimal"
        )
        clone_btn.click(status_working, inputs=None, outputs=[clone_status, design_btn, clone_btn, tts_btn], show_progress="hidden").then(
            generate_voice_clone,
            inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, save_cloned, use_cloned],
            outputs=[clone_audio_out, clone_status, design_btn, clone_btn, tts_btn],
            show_progress="minimal"
        )
        tts_btn.click(status_working, inputs=None, outputs=[tts_status, design_btn, clone_btn, tts_btn], show_progress="hidden").then(
            generate_custom_voice,
            inputs=[tts_text, tts_language, tts_speaker, tts_instruct],
            outputs=[tts_audio_out, tts_status, design_btn, clone_btn, tts_btn],
            show_progress="minimal"
        )

    demo.unload(fn=unload)


if __name__ == "__main__":
    demo.launch(show_api=False, )