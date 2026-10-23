"""
TalkBuddy — LLM service via external llama.cpp server HTTP API.
Streams tokens from POST /v1/chat/completions using Server-Sent Events (SSE).
Compatible with the OpenAI chat completions streaming format.
"""

import json
from typing import AsyncGenerator, Dict, List, Optional

import httpx

from app.core.config import LLMSettings
from app.core.logging import get_logger
from app.pipeline.interruption import CancellationToken

logger = get_logger()

# SSE data line prefix
_DATA_PREFIX = "data: "
_DONE_SENTINEL = "[DONE]"


class LLMService:
    """
    LLM inference service backed by an external llama.cpp server.
    Communicates via the OpenAI-compatible /v1/chat/completions SSE endpoint.
    The server handles model loading, GPU offloading, and KV-cache management.
    """

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self._client: Optional[httpx.AsyncClient] = None

    def initialize(self) -> None:
        """Create the persistent async HTTP client."""
        logger.info(f"LLM service connecting to external server: {self.settings.api_url}")
        self._client = httpx.AsyncClient(
            base_url=self.settings.api_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=120.0,   # long read for streaming generation
                write=10.0,
                pool=5.0,
            ),
        )
        logger.info("LLM HTTP client initialized")

    @property
    def is_loaded(self) -> bool:
        """Always true once the client is created (server manages the model)."""
        return self._client is not None

    async def check_server(self) -> bool:
        """Ping the llama.cpp /health endpoint. Returns True if reachable."""
        try:
            resp = await self._client.get("/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously stream tokens from the external llama.cpp server.
        Parses SSE lines and yields content deltas.
        Checks cancellation_token after every SSE event for instant barge-in.
        """
        if not self.is_loaded:
            raise RuntimeError("LLMService not initialized. Call initialize() first.")

        payload = {
            "messages": messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
            "stream": True,
        }

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    # Check barge-in first before processing each SSE line
                    if cancellation_token and cancellation_token.is_cancelled():
                        logger.debug("LLM generation cancelled by barge-in")
                        break

                    if not line.startswith(_DATA_PREFIX):
                        continue

                    data_str = line[len(_DATA_PREFIX):]
                    if data_str.strip() == _DONE_SENTINEL:
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse SSE chunk: {data_str!r}")
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token

        except httpx.ConnectError:
            logger.error(
                f"Cannot connect to llama.cpp server at {self.settings.api_url}. "
                "Start it with: ./llama-server -m <model.gguf> --host 0.0.0.0 --port 8080 -ngl -1"
            )
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"llama.cpp server returned error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise

    async def generate(
        self,
        messages: List[Dict[str, str]],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> str:
        """
        Generate a complete response (non-streaming convenience method).
        Collects all streamed tokens into a single string.
        """
        chunks: List[str] = []
        async for token in self.generate_stream(messages, cancellation_token):
            chunks.append(token)
        return "".join(chunks)

    async def shutdown(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client is not None:
            logger.info("Shutting down LLM HTTP client")
            await self._client.aclose()
            self._client = None
