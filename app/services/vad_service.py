"""
TalkBuddy — Silero VAD v5 state machine for speech detection.
Processes 512-sample (32ms) frames and detects speech start/end boundaries.
"""

from typing import Tuple

import numpy as np
import torch

from app.core.config import VADSettings
from app.core.constants import VAD_START, VAD_SPEAKING, VAD_END, VAD_SILENCE, VAD_FRAME_MS
from app.core.logging import get_logger

logger = get_logger()


class VADService:
    """
    Voice Activity Detection using Silero VAD v5.
    Runs a stateful recurrent model over 512-sample windows and
    detects speech start/end transitions using hysteresis thresholds.
    """

    def __init__(self, settings: VADSettings):
        self.settings = settings
        self._model = None
        self._triggered = False
        self._temp_end_ms = 0
        self._current_speech_ms = 0

    def initialize(self) -> None:
        """Load Silero VAD v5 model via torch hub."""
        logger.info("Loading Silero VAD v5 model")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=True,
            trust_repo=True,
        )
        self.reset_state()
        logger.info("VAD model loaded successfully")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def reset_state(self) -> None:
        """Reset internal recurrent states and endpoint tracking."""
        if self._model is not None:
            self._model.reset_states()
        self._triggered = False
        self._temp_end_ms = 0
        self._current_speech_ms = 0

    def process_frame(self, audio_chunk: np.ndarray) -> float:
        """
        Process a single 512-sample (32ms at 16kHz) float32 chunk.

        Args:
            audio_chunk: Float32 numpy array, 512 samples at 16kHz

        Returns:
            Speech probability (0.0 to 1.0)
        """
        if self._model is None:
            raise RuntimeError("VADService not initialized. Call initialize() first.")

        tensor = torch.from_numpy(audio_chunk).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        speech_prob = self._model(tensor, 16000).item()
        return speech_prob

    def detect_endpoint(self, speech_prob: float) -> Tuple[bool, str]:
        """
        Determine speech state transition using hysteresis logic.

        Args:
            speech_prob: Speech probability from process_frame()

        Returns:
            Tuple of (state_changed, state_name):
            - state_changed: True if a transition happened (START or END)
            - state_name: VAD_START, VAD_SPEAKING, VAD_END, or VAD_SILENCE
        """
        if speech_prob >= self.settings.threshold:
            # Speech detected — reset silence counter
            self._temp_end_ms = 0

            if not self._triggered:
                # Accumulate speech duration before triggering
                self._current_speech_ms += VAD_FRAME_MS
                if self._current_speech_ms >= self.settings.min_speech_duration_ms:
                    self._triggered = True
                    logger.debug(f"VAD: Speech START (prob={speech_prob:.2f})")
                    return True, VAD_START
                return False, VAD_SILENCE
            return False, VAD_SPEAKING
        else:
            # Silence detected
            if self._triggered:
                # Accumulate silence during active speech
                self._temp_end_ms += VAD_FRAME_MS
                if self._temp_end_ms >= self.settings.min_silence_duration_ms:
                    # Endpoint reached — speech turn is complete
                    self._triggered = False
                    self._temp_end_ms = 0
                    self._current_speech_ms = 0
                    logger.debug(f"VAD: Speech END (silence={self.settings.min_silence_duration_ms}ms)")
                    return True, VAD_END
                return False, VAD_SPEAKING
            else:
                # Idle silence
                self._current_speech_ms = 0
                return False, VAD_SILENCE

    def process_and_detect(self, audio_chunk: np.ndarray) -> Tuple[bool, str, float]:
        """
        Combined process + detect in one call.

        Args:
            audio_chunk: Float32 numpy array, 512 samples at 16kHz

        Returns:
            Tuple of (state_changed, state_name, speech_probability)
        """
        prob = self.process_frame(audio_chunk)
        changed, state = self.detect_endpoint(prob)
        return changed, state, prob

    def shutdown(self) -> None:
        """Release model resources."""
        if self._model is not None:
            logger.info("Shutting down VAD service")
            self._model = None
