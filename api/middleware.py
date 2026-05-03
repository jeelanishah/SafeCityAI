"""Custom middleware."""

import time
from typing import Callable

from fastapi import Request, Response
from loguru import logger

__all__ = ["setup_middleware"]


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log all requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - Time: {process_time:.4f}s"
    )
    return response


async def error_handler(request: Request, exc: Exception) -> Response:
    """Handle errors."""
    logger.error(f"Error: {exc}")
    return Response("Internal Server Error", status_code=500)


def setup_middleware(app) -> None:
    """Setup all middleware."""
    app.middleware("http")(logging_middleware)
