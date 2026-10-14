"""
TalkBuddy — Pydantic BaseSettings configuration.
All settings overridable via .env (nested delimiter: __).
"""

from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioSettings(BaseSettings):
    """Audio format settings."""
    input_sample_rate: int = Field(default=16000, description="Input PCM sample rate in Hz")
    output_sample_rate: int = Field(default=24000, description="TTS output sample rate in Hz")
    channels: int = Field(default=1, description="Mono audio")
    sample_width: int = Field(default=2, description="16-bit PCM (2 bytes)")
    chunk_size_samples: int = Field(default=512, description="VAD frame chunk size")


class VADSettings(BaseSettings):
    """Voice Activity Detection settings."""
    model_name: str = Field(default="silero_vad", description="VAD engine")
    threshold: float = Field(default=0.5, description="Speech probability threshold (0.0-1.0)")
    min_speech_duration_ms: int = Field(default=96, description="Min speech before triggering (ms)")
    min_silence_duration_ms: int = Field(default=350, description="Silence to mark end of utterance (ms)")
    speech_pad_ms: int = Field(default=60, description="Padding added to audio bounds (ms)")


class STTSettings(BaseSettings):
    """Speech-to-Text settings (faster-whisper)."""
    model_size: str = Field(default="large-v3-turbo", description="faster-whisper model tag/path")
    device: Literal["cpu", "cuda", "auto"] = Field(default="cpu", description="CTranslate2 device")
    compute_type: Literal["default", "int8", "float16"] = Field(default="int8", description="Quantization")
    beam_size: int = Field(default=1, description="Greedy search for minimum latency")
    language: str = Field(default="en", description="Target transcription language")


class LLMSettings(BaseSettings):
    """LLM settings (llama-cpp-python with Vulkan)."""
    model_path: str = Field(default="models/llm/model.gguf", description="Path to GGUF model file")
    n_gpu_layers: int = Field(default=-1, description="GPU layers offloaded to Vulkan (-1 for all)")
    n_ctx: int = Field(default=4096, description="Context window size")
    n_threads: int = Field(default=8, description="CPU threads for non-offloaded layers")
    n_batch: int = Field(default=512, description="Batch size for prompt processing")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling")
    max_tokens: int = Field(default=256, description="Max generated tokens per turn")
    system_prompt: str = Field(
        default="You are TalkBuddy, a friendly and concise real-time voice assistant. "
                "Keep your answers conversational, natural, and brief (1-3 sentences).",
        description="System prompt"
    )


class TTSSettings(BaseSettings):
    """Text-to-Speech settings (Kokoro-82M)."""
    model_name: str = Field(default="hexgrad/Kokoro-82M", description="Kokoro model ID")
    voice: str = Field(default="af_heart", description="Default Kokoro voice identifier")
    speed: float = Field(default=1.0, description="Speaking rate multiplier")
    lang_code: str = Field(default="a", description="Language code (a for American English)")


class ServerSettings(BaseSettings):
    """FastAPI server settings."""
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")
    reload: bool = Field(default=False, description="Auto-reload for dev")
    log_level: str = Field(default="INFO", description="Logging level")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")


class TalkBuddyConfig(BaseSettings):
    """Root configuration for TalkBuddy."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)


# Global settings singleton
settings = TalkBuddyConfig()
