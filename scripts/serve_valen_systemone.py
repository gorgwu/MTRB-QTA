"""Expose a local Valen model through the TypeSafe System One HTTP contract."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModel


class SystemOneRequest(BaseModel):
    model: str = "valen-4b"
    state: Any
    questions: dict[str, dict[str, Any]] = Field(min_length=1)


def _valen_state(state: Any) -> str | dict[str, Any]:
    """Convert JiuwenSwarm's state object to Valen's messages-based schema."""
    if isinstance(state, str):
        return state
    if not isinstance(state, dict):
        raise HTTPException(status_code=422, detail="state must be text or an object")
    if isinstance(state.get("messages"), list):
        return state

    query = str(state.get("user_request") or "").strip()
    conversation = state.get("conversation") or []
    if not isinstance(conversation, list):
        raise HTTPException(status_code=422, detail="state.conversation must be a list")
    messages = []
    for item in conversation:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower()
        content = item.get("content")
        if role in {"user", "assistant", "system"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    if query and not any(m["role"] == "user" and m["content"] == query for m in messages):
        messages.append({"role": "user", "content": query})
    if not messages:
        raise HTTPException(status_code=422, detail="state has no user request or messages")
    return {"messages": messages}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_id = os.getenv("VALEN_MODEL", "Valen-Team/Valen-4B")
    device = os.getenv("VALEN_DEVICE", "cuda")
    dtype_name = os.getenv("VALEN_DTYPE", "auto").lower()
    attention = os.getenv("VALEN_ATTN", "sdpa")
    dtype = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise RuntimeError("VALEN_DTYPE must be auto, bf16, fp16, or fp32")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("VALEN_DEVICE requests CUDA, but CUDA is unavailable")

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation=attention,
    ).to(device).eval()
    app.state.model = model
    app.state.model_id = model_id
    app.state.device = device
    app.state.inference_lock = asyncio.Lock()
    yield
    del app.state.model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


app = FastAPI(title="Valen System One adapter", lifespan=lifespan)
logger = logging.getLogger("valen-systemone")


@app.middleware("http")
async def authenticate_and_time(request: Request, call_next):
    started = time.perf_counter()
    expected_key = os.getenv("VALEN_API_KEY", "")
    if (
        expected_key
        and request.url.path.startswith("/v1")
        and not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {expected_key}"
        )
    ):
        response = JSONResponse(
            {"detail": "missing or invalid API key"},
            status_code=401,
            headers={"www-authenticate": "Bearer"},
        )
    else:
        response = await call_next(request)
    response.headers["x-typesafe-request-id"] = (
        request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
    )
    response.headers["server-timing"] = (
        f"app;dur={(time.perf_counter() - started) * 1000:.1f}"
    )
    return response


@app.get("/healthz")
async def health(request: Request):
    model = getattr(request.app.state, "model", None)
    return {
        "ready": model is not None,
        "model": getattr(request.app.state, "model_id", None),
        "device": getattr(request.app.state, "device", None),
    }


@app.get("/v1/models")
async def models(request: Request):
    return {"models": [{"name": "valen-4b", "id": request.app.state.model_id}]}


@app.post("/v1/systemone")
async def systemone(payload: SystemOneRequest, request: Request):
    model = request.app.state.model
    body = {
        "state": _valen_state(payload.state),
        "questions": payload.questions,
    }
    try:
        async with request.app.state.inference_lock:
            result = await asyncio.to_thread(_predict, model, body)
    except HTTPException:
        raise
    except Exception as exc:
        # Keep the reason in the server log, while returning an API-shaped error.
        request_id = uuid.uuid4().hex
        logger.exception("Valen inference failed; request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Valen inference failed; request_id={request_id}: {type(exc).__name__}: {exc}",
        ) from exc
    return result


def _predict(model, body: dict[str, Any]) -> dict[str, Any]:
    with torch.inference_mode():
        return model.predict(body, execution="shared_state", temperature=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--model", default=os.getenv("VALEN_MODEL", "Valen-Team/Valen-4B"))
    parser.add_argument("--device", default=os.getenv("VALEN_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.getenv("VALEN_DTYPE", "auto"))
    parser.add_argument("--attention", default=os.getenv("VALEN_ATTN", "sdpa"))
    args = parser.parse_args()
    os.environ["VALEN_MODEL"] = args.model
    os.environ["VALEN_DEVICE"] = args.device
    os.environ["VALEN_DTYPE"] = args.dtype
    os.environ["VALEN_ATTN"] = args.attention
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
