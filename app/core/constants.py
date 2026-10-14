"""
TalkBuddy — Audio sample rates, frame sizes, VAD constants.
"""

# --- Audio Formats ---
INPUT_SAMPLE_RATE = 16000       # 16kHz input from client microphone
OUTPUT_SAMPLE_RATE = 24000      # 24kHz output from Kokoro TTS
CHANNELS = 1                    # Mono audio throughout
SAMPLE_WIDTH = 2                # 16-bit PCM (2 bytes per sample)

# --- VAD Frame Sizing ---
VAD_FRAME_SAMPLES = 512         # Silero VAD processes 512 samples per frame
VAD_FRAME_MS = 32               # 512 samples at 16kHz = 32ms per frame

# --- Pipeline Defaults ---
SENTENCE_MIN_FIRST_CHARS = 15   # Flush first sentence chunk early for low TTFA
SENTENCE_MIN_NORMAL_CHARS = 35  # Subsequent sentences need more context for prosody

# --- Session ---
SESSION_TTL_SECONDS = 3600      # 1 hour idle timeout
SESSION_MAX_HISTORY = 25        # Max messages kept in context window

# --- Agent States ---
STATE_LISTENING = "LISTENING"
STATE_THINKING = "THINKING"
STATE_SPEAKING = "SPEAKING"

# --- VAD States ---
VAD_START = "START"
VAD_SPEAKING = "SPEAKING"
VAD_END = "END"
VAD_SILENCE = "SILENCE"
