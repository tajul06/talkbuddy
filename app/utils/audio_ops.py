"""
TalkBuddy — Resampling, PCM/float32 conversions, WAV packing utilities.
"""

import io
import struct
from typing import Union

import numpy as np
import soundfile as sf
from scipy import signal

from app.core.constants import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE


def pcm16_to_float32(pcm_data: bytes) -> np.ndarray:
    """
    Convert 16-bit PCM bytes to float32 numpy array [-1.0, 1.0].

    Args:
        pcm_data: Raw 16-bit PCM bytes (little-endian signed integers)

    Returns:
        Float32 numpy array normalized to [-1.0, 1.0]
    """
    # Unpack as int16
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    # Normalize to [-1.0, 1.0]
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """
    Convert float32 numpy array to 16-bit PCM bytes.

    Args:
        audio: Float32 numpy array in range [-1.0, 1.0]

    Returns:
        Raw 16-bit PCM bytes (little-endian signed integers)
    """
    # Clip to [-1.0, 1.0] and scale to int16 range
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to a different sample rate using high-quality polyphase filtering.

    Args:
        audio: Float32 audio array
        orig_sr: Original sample rate (Hz)
        target_sr: Target sample rate (Hz)

    Returns:
        Resampled float32 audio array
    """
    if orig_sr == target_sr:
        return audio

    # Use scipy's polyphase resampler for high quality
    num_samples = int(len(audio) * target_sr / orig_sr)
    return signal.resample(audio, num_samples).astype(np.float32)


def pack_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Pack float32 audio into a WAV file bytes (in-memory).

    Args:
        audio: Float32 numpy array
        sample_rate: Sample rate in Hz

    Returns:
        Complete WAV file as bytes (including header)
    """
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def unpack_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Unpack a WAV file bytes into float32 audio array and sample rate.

    Args:
        wav_bytes: Complete WAV file bytes (including header)

    Returns:
        Tuple of (float32 audio array, sample_rate)
    """
    buffer = io.BytesIO(wav_bytes)
    audio, sr = sf.read(buffer, dtype="float32")
    return audio, sr


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo/multi-channel audio to mono by averaging channels.

    Args:
        audio: Audio array (shape: (samples,) or (samples, channels))

    Returns:
        Mono audio array (shape: (samples,))
    """
    if audio.ndim == 1:
        return audio
    # Average across channels
    return audio.mean(axis=1)


def normalize_audio_level(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio to a target peak level.

    Args:
        audio: Float32 audio array
        target_peak: Target peak amplitude (0.0 to 1.0)

    Returns:
        Normalized audio array
    """
    peak = np.abs(audio).max()
    if peak > 0:
        return audio * (target_peak / peak)
    return audio


def prepare_input_audio(audio_bytes: Union[bytes, np.ndarray], is_raw_pcm: bool = False) -> np.ndarray:
    """
    Prepare input audio for VAD/STT pipeline:
    - Unpack WAV or convert raw PCM to float32
    - Ensure mono
    - Resample to INPUT_SAMPLE_RATE (16kHz)

    Args:
        audio_bytes: Either WAV file bytes or raw 16-bit PCM bytes
        is_raw_pcm: True if audio_bytes is raw PCM (no header)

    Returns:
        Float32 mono audio at 16kHz
    """
    if isinstance(audio_bytes, np.ndarray):
        audio = audio_bytes
        sr = INPUT_SAMPLE_RATE
    elif is_raw_pcm:
        audio = pcm16_to_float32(audio_bytes)
        sr = INPUT_SAMPLE_RATE
    else:
        audio, sr = unpack_wav(audio_bytes)

    audio = ensure_mono(audio)
    audio = resample_audio(audio, sr, INPUT_SAMPLE_RATE)
    return audio


def prepare_output_audio(audio: np.ndarray) -> bytes:
    """
    Prepare TTS output audio for delivery:
    - Ensure it's at OUTPUT_SAMPLE_RATE (24kHz)
    - Pack as WAV bytes

    Args:
        audio: Float32 audio array (assumed to be at OUTPUT_SAMPLE_RATE)

    Returns:
        WAV file bytes
    """
    return pack_wav(audio, OUTPUT_SAMPLE_RATE)
