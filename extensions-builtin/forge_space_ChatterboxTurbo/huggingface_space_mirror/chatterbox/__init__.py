try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = "0.1.4"


from .tts import ChatterboxTTS
