# TalkBuddy

**A real-time conversational voice agent powered by local open-source AI models.**

TalkBuddy delivers ultra-low-latency voice conversations (~3 seconds time-to-first-audio) using GPU-accelerated inference on AMD Radeon via Vulkan. Inspired by Sesame Maya, it runs entirely on your hardware with no cloud dependencies.

---

## Features

- **Real-time streaming** — Full-duplex WebSocket with <3s TTFA (Time-To-First-Audio)
- **Live barge-in** — Interrupt the agent mid-sentence naturally
- **GPU acceleration** — Vulkan offloading for LLM and STT on AMD/NVIDIA
- **Modern web UI** — React + Tailwind testing dashboard with live audio visualization
- **100% local** — All models run on your machine, no API keys, no cloud
- **Model-agnostic** — Load any GGUF LLM and switch STT/TTS engines easily

---

## Performance

**Hardware:** AMD Ryzen 5 3600 (6-core/12-thread) + AMD Radeon RX 5600 XT (6GB VRAM)

| Metric | Time |
|--------|------|
| **STT (Whisper small GPU)** | 260-450ms |
| **LLM First Token (Llama 3.2 3B)** | 620-950ms |
| **TTS (Kokoro-82M)** | 100-180ms |
| **Total TTFA** | **2.8-3.5 seconds** |

---

## Architecture

```
[Microphone] → [VAD] → [Whisper.cpp GPU] → [Llama 3.2 3B GPU] → [Kokoro TTS] → [Speaker]
```

- **VAD**: Silero VAD v5 (speech endpoint detection)
- **STT**: whisper.cpp with Vulkan (GPU) or faster-whisper (CPU)
- **LLM**: External llama-server (Vulkan GPU offloading)
- **TTS**: Kokoro-82M (24kHz, multi-voice)
- **Frontend**: React + TypeScript + Tailwind CSS

---

## Quick Start

### Prerequisites

- **Windows 11** (with MSYS2 for building C++ components)
- **Python 3.11+**
- **Node.js 18+** (for frontend)
- **GPU**: AMD Radeon (Vulkan) or NVIDIA (CUDA)
- **Disk Space**: ~8GB (models + dependencies)

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd talkbuddy
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install with uv (recommended)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### 3. Download Models

#### LLM Model (Required)
```bash
# Llama 3.2 3B Instruct Q4_K_M (~2GB)
curl -L -o models/llm/llama-3.2-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

#### Whisper Model (Required for GPU STT)
```bash
# Whisper small model (~466MB)
curl -L -o models/whisper/ggml-small.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
```

**Note:** Kokoro TTS, Silero VAD, and faster-whisper models auto-download on first use.

### 4. Build whisper.cpp with Vulkan (GPU STT)

**In MSYS2 UCRT64 terminal:**
```bash
cd ~
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
mkdir build && cd build
cmake .. -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### 5. Build llama.cpp with Vulkan (GPU LLM)

**In MSYS2 UCRT64 terminal:**
```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### 6. Configure Environment

```bash
cp .env.example .env
# Edit .env to customize settings
```

**Key settings:**
```bash
STT__BACKEND=whisper.cpp       # or "faster-whisper" for CPU
STT__MODEL_SIZE=small
LLM__MAX_TOKENS=128            # Shorter responses = faster
VAD__MIN_SILENCE_DURATION_MS=200
```

### 7. Start Services

**Terminal 1 — LLM Server:**
```bash
C:/msys64/home/RISEBURG/llama.cpp/build/bin/llama-server.exe \
  -m F:/talkbuddy/models/llm/llama-3.2-3b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 4096 -fa on --cache-prompt
```

**Terminal 2 — Whisper STT Server (if using whisper.cpp):**
```bash
C:/msys64/home/RISEBURG/whisper.cpp/build/bin/whisper-server.exe \
  -m F:/talkbuddy/models/whisper/ggml-small.bin \
  --host 0.0.0.0 --port 8081 \
  -t 4 -l en -fa
```

**Terminal 3 — TalkBuddy Backend:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 4 — Web UI:**
```bash
cd talkbuddy-ui
npm install  # First time only
npm run dev
```

### 8. Open Web Interface

Open your browser to:
- **Local:** http://localhost:5174
- **LAN:** http://192.168.0.151:5174 (replace with your IP)

---

## API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | System status & model info |
| `POST` | `/api/v1/voice/chat` | Upload WAV → get WAV response |
| `POST` | `/api/v1/text/chat` | Text → text response |
| `POST` | `/api/v1/text/chat/stream` | Text → streamed audio |
| `POST` | `/api/v1/sessions` | Create session |
| `GET` | `/api/v1/sessions` | List sessions |
| `DELETE` | `/api/v1/sessions/{id}` | Delete session |

### WebSocket

**Endpoint:** `ws://localhost:8000/api/v1/voice/stream`

**Client → Server:**
- Binary frames: 16kHz 16-bit mono PCM (512 samples/frame)
- JSON control: `{"type": "init", "voice": "af_heart"}`, `{"type": "interrupt"}`

**Server → Client:**
- Binary frames: 24kHz 16-bit mono PCM
- JSON events: `agent_state`, `vad_state`, `transcript`, `llm_chunk`, `turn_end`, `interrupted`

---

## Available Voices

**Female (American English):**
- `af_heart` — Warm, friendly (default)
- `af_sky` — Calm, gentle
- `af_bella` — Confident, professional
- `af_sarah` — Young, energetic

**Male:**
- `am_adam` — Deep, authoritative
- `am_michael` — Friendly, casual

**British:**
- `bf_emma` — British female
- `bm_george` — British male

---

## Testing

### Unit Tests
```bash
python -m pytest tests/unit/ -v
```

### Manual Testing with curl

**Upload audio file:**
```bash
curl -X POST http://localhost:8000/api/v1/voice/chat \
  -F "audio=@recording.wav" \
  -F "voice=af_heart" \
  --output response.wav
```

---

## Configuration

All settings configurable via `.env` with `__` delimiter:

```bash
# Server
SERVER__HOST=0.0.0.0
SERVER__PORT=8000

# STT
STT__BACKEND=whisper.cpp      # "whisper.cpp" or "faster-whisper"
STT__MODEL_SIZE=small
STT__LANGUAGE=en

# LLM
LLM__API_URL=http://localhost:8080
LLM__TEMPERATURE=0.7
LLM__MAX_TOKENS=128

# TTS
TTS__VOICE=af_heart
TTS__SPEED=1.1

# VAD
VAD__THRESHOLD=0.5
VAD__MIN_SILENCE_DURATION_MS=200
```

---

## Troubleshooting

### GPU not detected
- Ensure Vulkan SDK is installed
- Check GPU with: `vulkaninfo | grep deviceName`
- Rebuild llama.cpp/whisper.cpp with `-DGGML_VULKAN=1`

### STT too slow
- Switch to `STT__BACKEND=whisper.cpp` for GPU acceleration
- Or use smaller model: `STT__MODEL_SIZE=tiny`

### LLM too slow
- Use smaller model (3B instead of 7B/8B)
- Reduce `LLM__MAX_TOKENS` to 64 or 128
- Enable Flash Attention: `-fa on`

### Audio issues on LAN
- Chrome requires HTTPS for microphone access over network
- Workaround: Flag the origin as safe in `chrome://flags/#unsafely-treat-insecure-origin-as-secure`

---

## Project Structure

```
talkbuddy/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── core/                      # Config, logging, constants
│   ├── services/                  # VAD, STT, LLM, TTS services
│   ├── pipeline/                  # Orchestrator, chunker, interruption
│   ├── api/v1/                    # REST + WebSocket endpoints
│   └── utils/                     # Audio ops, executors
├── models/
│   ├── llm/                       # Place GGUF models here
│   └── whisper/                   # Place whisper models here
├── talkbuddy-ui/                  # React frontend
├── tests/
│   └── unit/                      # Unit tests
├── .env.example                   # Environment template
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Vulkan-accelerated LLM inference
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - GPU STT
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CPU STT
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - High-quality TTS
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection

---

**Built with care for the open-source AI community**
