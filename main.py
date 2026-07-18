from __future__ import annotations

from azure.ai.agentserver.invocations import InvocationAgentServerHost
from starlette.requests import Request
from starlette.responses import JSONResponse

from taslow_email_extraction_agent.hosted_agent import process_email_invocation

app = InvocationAgentServerHost()


@app.invoke_handler
async def handle_invoke(request: Request) -> JSONResponse:
    correlation_id = request.headers.get("x-correlation-id")
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse(
            {
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "The invocation body must be valid JSON.",
                    "correlationId": correlation_id,
                }
            },
            status_code=400,
        )

    status_code, body = await process_email_invocation(payload, correlation_id=correlation_id)
    return JSONResponse(body, status_code=status_code)


if __name__ == "__main__":
    app.run()
