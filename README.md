# TalkBuddy

A real-time conversational voice agent powered by local AI models — ultra-low latency, barge-in support, fully open-source.

## Features

- 🎙️ **Real-time voice conversation** with sub-second response times
- 🚀 **Local inference only** — no cloud APIs, complete privacy
- 🔊 **Natural speech synthesis** via Kokoro-82M (82M parameters, CPU-friendly)
- 🎯 **Barge-in support** — interrupt the agent mid-sentence
- 🔄 **Async pipeline** — FastAPI with WebSocket streaming
- 🖥️ **AMD Vulkan acceleration** — llama.cpp with Vulkan backend

## Tech Stack

- **Backend**: FastAPI (async REST + WebSocket)
- **LLM**: llama-cpp-python (Vulkan) — model-agnostic GGUF loader
- **STT**: faster-whisper (large-v3-turbo)
- **TTS**: Kokoro-82M
- **VAD**: Silero VAD v5

## Quick Start

1. **Clone and setup**
```bash
cd F:\talkbuddy
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Add your LLM model**
Place a GGUF model file at `models/llm/model.gguf` (or set `LLM__MODEL_PATH` in `.env`)

4. **Run the server**
```bash
uvicorn app.main:app --reload
```

5. **Test health endpoint**
```bash
curl http://localhost:8000/api/v1/health
```

## API Endpoints

- `GET /api/v1/health` — System status and GPU info
- `POST /api/v1/voice/chat` — Upload audio, get audio response
- `POST /api/v1/text/chat` — Text in, text/audio out
- `WS /api/v1/voice/stream` — Real-time bidirectional audio streaming

## Architecture

```
[Mic] → [VAD] → [STT] → [LLM] → [TTS] → [Speaker]
         ↓        ↓       ↓        ↓
    Speech    Text    Tokens   Audio
    Detection          Stream   Chunks
```

## License

MIT
