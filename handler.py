"""RunPod serverless handler stub.

This image runs as a load-balancer worker (handler_lb.py / FastAPI on port 80).
The RunPod serverless SDK is not active, but this module satisfies the required
handler.py structure for RunPod Hub validation and tooling.

To send requests, use the OpenAI-compatible endpoints:
  POST /v1/chat/completions
  POST /v1/completions
  POST /v1/messages  (Anthropic)
"""


async def handler(event):
    return {
        "error": (
            "This endpoint runs as a load-balancer worker. "
            "Send requests directly to /v1/chat/completions."
        )
    }
