"""RunPod load-balancer worker — vLLM with native OpenAI-compatible API.

Runs a FastAPI/uvicorn HTTP server on port 80. RunPod's load balancer polls
/ping to discover and route to healthy workers:
  - 204: initializing (not yet in rotation)
  - 200: ready (accepting traffic)

Exposed endpoints:
  GET  /ping
  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/completions
  POST /v1/responses
  GET  /v1/responses/{id}
  POST /v1/responses/{id}/cancel
  POST /v1/messages  (Anthropic)

Engine args are fully configurable from environment variables via engine_args.py.
MODEL_PATH (or MODEL_NAME) must be provided at runtime.
"""

import logging
import multiprocessing
import os
import sys
import traceback
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.entrypoints.anthropic.protocol import (
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest, ResponsesResponse
from vllm.entrypoints.serve.render.serving import OpenAIServingRender

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_is_ready = False
_chat_engine = None
_completion_engine = None
_responses_engine = None
_messages_engine = None
_serving_models = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _is_ready, _chat_engine, _completion_engine, _responses_engine, _messages_engine, _serving_models

    try:
        from engine import vLLMEngine
        from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        log.info("Initializing vLLM engine...")
        vllm_engine = vLLMEngine()
        engine_args = vllm_engine.engine_args
        llm = vllm_engine.llm

        served_model_name = (
            os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE")
            or engine_args.served_model_name
            or engine_args.model
        )

        _serving_models = OpenAIServingModels(
            engine_client=llm,
            base_model_paths=[BaseModelPath(name=served_model_name, model_path=engine_args.model)],
            lora_modules=None,
        )
        await _serving_models.init_static_loras()

        chat_template = None
        if vllm_engine.tokenizer and hasattr(vllm_engine.tokenizer, "tokenizer"):
            chat_template = vllm_engine.tokenizer.tokenizer.chat_template

        _serving_render = OpenAIServingRender(
            model_config=llm.model_config,
            renderer=None,
            io_processor=None,
            model_registry=_serving_models.registry,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            trust_request_chat_template=os.getenv("TRUST_REQUEST_CHAT_TEMPLATE", "false").lower() == "true",
            enable_auto_tools=os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").lower() == "true",
            exclude_tools_when_tool_choice_none=os.getenv("EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE", "false").lower() == "true",
            tool_parser=os.getenv("TOOL_CALL_PARSER", "") or None,
            log_error_stack=False,
        )

        _chat_engine = OpenAIServingChat(
            engine_client=llm,
            models=_serving_models,
            response_role=os.getenv("OPENAI_RESPONSE_ROLE", "assistant"),
            openai_serving_render=_serving_render,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            trust_request_chat_template=os.getenv("TRUST_REQUEST_CHAT_TEMPLATE", "false").lower() == "true",
            return_tokens_as_token_ids=os.getenv("RETURN_TOKENS_AS_TOKEN_IDS", "false").lower() == "true",
            reasoning_parser=os.getenv("REASONING_PARSER", "") or "",
            enable_auto_tools=os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").lower() == "true",
            exclude_tools_when_tool_choice_none=os.getenv("EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE", "false").lower() == "true",
            tool_parser=os.getenv("TOOL_CALL_PARSER", "") or None,
            enable_prompt_tokens_details=os.getenv("ENABLE_PROMPT_TOKENS_DETAILS", "false").lower() == "true",
            enable_force_include_usage=os.getenv("ENABLE_FORCE_INCLUDE_USAGE", "false").lower() == "true",
            enable_log_outputs=os.getenv("ENABLE_LOG_OUTPUTS", "false").lower() == "true",
        )

        _completion_engine = OpenAIServingCompletion(
            engine_client=llm,
            models=_serving_models,
            openai_serving_render=_serving_render,
            request_logger=None,
            return_tokens_as_token_ids=os.getenv("RETURN_TOKENS_AS_TOKEN_IDS", "false").lower() == "true",
            enable_prompt_tokens_details=os.getenv("ENABLE_PROMPT_TOKENS_DETAILS", "false").lower() == "true",
            enable_force_include_usage=os.getenv("ENABLE_FORCE_INCLUDE_USAGE", "false").lower() == "true",
        )

        _responses_engine = OpenAIServingResponses(
            engine_client=llm,
            models=_serving_models,
            openai_serving_render=_serving_render,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            return_tokens_as_token_ids=os.getenv("RETURN_TOKENS_AS_TOKEN_IDS", "false").lower() == "true",
            reasoning_parser=os.getenv("REASONING_PARSER", "") or "",
            enable_auto_tools=os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").lower() == "true",
            tool_parser=os.getenv("TOOL_CALL_PARSER", "") or None,
            tool_server=None,
            enable_prompt_tokens_details=os.getenv("ENABLE_PROMPT_TOKENS_DETAILS", "false").lower() == "true",
            enable_force_include_usage=os.getenv("ENABLE_FORCE_INCLUDE_USAGE", "false").lower() == "true",
            enable_log_outputs=os.getenv("ENABLE_LOG_OUTPUTS", "false").lower() == "true",
        )

        _messages_engine = AnthropicServingMessages(
            engine_client=llm,
            models=_serving_models,
            response_role=os.getenv("OPENAI_RESPONSE_ROLE", "assistant"),
            openai_serving_render=_serving_render,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            return_tokens_as_token_ids=os.getenv("RETURN_TOKENS_AS_TOKEN_IDS", "false").lower() == "true",
            reasoning_parser=os.getenv("REASONING_PARSER", "") or "",
            enable_auto_tools=os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").lower() == "true",
            tool_parser=os.getenv("TOOL_CALL_PARSER", "") or None,
            enable_prompt_tokens_details=os.getenv("ENABLE_PROMPT_TOKENS_DETAILS", "false").lower() == "true",
            enable_force_include_usage=os.getenv("ENABLE_FORCE_INCLUDE_USAGE", "false").lower() == "true",
        )

        _is_ready = True
        log.info("vLLM load-balancer worker ready")

    except Exception as e:
        log.error("Startup failed: %s\n%s", e, traceback.format_exc())
        sys.exit(1)

    yield  # serve requests


app = FastAPI(title="vLLM Load Balancer Worker", lifespan=lifespan)


@app.get("/ping")
async def ping():
    """RunPod load-balancer health check. 204 = loading, 200 = ready."""
    if not _is_ready or not all([_chat_engine, _completion_engine, _responses_engine, _messages_engine, _serving_models]):
        return Response(status_code=204)
    return Response(status_code=200)


@app.get("/v1/models")
async def list_models():
    models = await _serving_models.show_available_models()
    return JSONResponse(models.model_dump())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    try:
        req = ChatCompletionRequest(**body)
    except Exception as e:
        return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error"}}, status_code=422)

    response = await _chat_engine.create_chat_completion(req, raw_request=request)

    if isinstance(response, ErrorResponse):
        return JSONResponse(response.model_dump(), status_code=response.error.code)
    if isinstance(response, ChatCompletionResponse):
        return JSONResponse(response.model_dump())

    async def event_stream():
        async for chunk in response:
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    try:
        req = CompletionRequest(**body)
    except Exception as e:
        return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error"}}, status_code=422)

    response = await _completion_engine.create_completion(req, raw_request=request)

    if isinstance(response, ErrorResponse):
        return JSONResponse(response.model_dump(), status_code=response.error.code)
    if isinstance(response, CompletionResponse):
        return JSONResponse(response.model_dump())

    async def event_stream():
        async for chunk in response:
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/responses")
async def create_responses(request: Request):
    body = await request.json()
    try:
        req = ResponsesRequest(**body)
    except Exception as e:
        return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error"}}, status_code=422)

    response = await _responses_engine.create_responses(req, raw_request=request)

    if isinstance(response, ErrorResponse):
        return JSONResponse(response.model_dump(), status_code=response.error.code)
    if isinstance(response, ResponsesResponse):
        return JSONResponse(response.model_dump())

    async def event_stream():
        async for event in response:
            event_type = getattr(event, "type", "unknown")
            yield f"event: {event_type}\ndata: {event.model_dump_json(indent=None)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/responses/{response_id}")
async def retrieve_responses(response_id: str, request: Request, starting_after: int | None = None, stream: bool | None = False):
    try:
        response = await _responses_engine.retrieve_responses(response_id, starting_after=starting_after, stream=stream)
    except Exception as e:
        return JSONResponse({"error": {"type": "invalid_request_error", "message": str(e)}}, status_code=422)

    if isinstance(response, ErrorResponse):
        return JSONResponse(response.model_dump(), status_code=response.error.code)
    if isinstance(response, ResponsesResponse):
        return JSONResponse(response.model_dump())

    async def event_stream():
        async for event in response:
            event_type = getattr(event, "type", "unknown")
            yield f"event: {event_type}\ndata: {event.model_dump_json(indent=None)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_responses(response_id: str, request: Request):
    try:
        response = await _responses_engine.cancel_responses(response_id)
    except Exception as e:
        return JSONResponse({"error": {"type": "invalid_request_error", "message": str(e)}}, status_code=422)

    if isinstance(response, ErrorResponse):
        return JSONResponse(response.model_dump(), status_code=response.error.code)
    return JSONResponse(response.model_dump())


@app.post("/v1/messages")
async def create_messages(request: Request):
    body = await request.json()
    try:
        req = AnthropicMessagesRequest(**body)
    except Exception as e:
        return JSONResponse({"error": {"type": "invalid_request_error", "message": str(e)}}, status_code=422)

    response = await _messages_engine.create_messages(req, raw_request=request)

    if isinstance(response, ErrorResponse):
        return JSONResponse(
            AnthropicErrorResponse(error=AnthropicError(type=response.error.type, message=response.error.message)).model_dump(),
            status_code=response.error.code,
        )
    if isinstance(response, AnthropicMessagesResponse):
        return JSONResponse(response.model_dump(exclude_none=True))

    return StreamingResponse(response, media_type="text/event-stream")


if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
