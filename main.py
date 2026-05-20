import os
import sys
import uvicorn
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from starlette.types import ASGIApp, Scope, Receive, Send

# Add current directory to python path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Pure ASGI Middleware to inject streaming headers without buffering.
# This avoids using FastAPI's `@app.middleware("http")` (BaseHTTPMiddleware)
# which is known to buffer StreamingResponse and prevent real-time SSE output.
class StreamingHeadersMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                is_sse = False
                for name, value in headers:
                    if name.lower() == b"content-type" and b"text/event-stream" in value:
                        is_sse = True
                        break
                
                if is_sse:
                    # Clean up existing headers that we want to control
                    filtered_headers = []
                    for name, value in headers:
                        lname = name.lower()
                        if lname in (b"cache-control", b"connection", b"pragma", b"content-length", b"x-accel-buffering"):
                            continue
                        filtered_headers.append((name, value))
                    
                    # Append optimization headers for SSE streaming on Google Front End (GFE) & reverse proxies
                    filtered_headers.append((b"x-accel-buffering", b"no"))
                    filtered_headers.append((b"cache-control", b"no-cache, no-transform"))
                    filtered_headers.append((b"connection", b"keep-alive"))
                    filtered_headers.append((b"pragma", b"no-cache"))
                    
                    message["headers"] = filtered_headers
            
            await send(message)

        await self.app(scope, receive, send_wrapper)

def create_app() -> FastAPI:
    print(f"Initializing ADK web server from directory: {current_dir}")
    # Pasamos allow_origins=["*"] para activar de forma nativa el middleware de CORS en ADK
    app = get_fast_api_app(agents_dir=current_dir, web=True, allow_origins=["*"])
    
    # Add pure ASGI middleware to the FastAPI app
    app.add_middleware(StreamingHeadersMiddleware)
    
    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting custom ADK server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

