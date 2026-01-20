"""Response Compression Middleware for FastAPI.

PERF-001 FIX: Implements gzip and brotli compression to reduce bandwidth usage
and improve API response times.

Expected Impact: 70-80% bandwidth reduction on text-based API responses

Configuration:
    - Minimum response size: 500 bytes (smaller responses sent uncompressed)
    - Compression level: 6 (balanced between speed and compression ratio)
    - Supported formats: gzip (universal), brotli (modern browsers)

Usage:
    # Middleware is automatically registered in app/main.py
    # Configure via environment variables:
    # ENABLE_COMPRESSION=true (default: true)
    # COMPRESSION_LEVEL=6 (default: 6, range: 1-9)
    # COMPRESSION_MIN_SIZE=500 (default: 500 bytes)
"""

import gzip
import io
import logging
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware that compresses responses using gzip or brotli.

    Compresses responses based on Accept-Encoding header and response size.
    Skips compression for small responses, already compressed content, and
    when client doesn't support compression.
    """

    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int = 500,
        gzip_level: int = 6,
        brotli_level: int = 4,
    ) -> None:
        """Initialize compression middleware.

        Args:
            app: ASGI application
            minimum_size: Minimum response size in bytes to compress
            gzip_level: Gzip compression level (1-9, default 6)
            brotli_level: Brotli compression level (0-11, default 4)

        """
        super().__init__(app)
        self.minimum_size = minimum_size
        self.gzip_level = gzip_level
        self.brotli_level = brotli_level

        # Check if brotli is available
        try:
            import brotli

            self.brotli_available = True
        except ImportError:
            self.brotli_available = False
            logger.debug("Brotli compression not available. Install with: pip install brotli")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request and compress response if applicable.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/route handler

        Returns:
            Response, potentially compressed

        """
        # Get client's accepted encodings
        accept_encoding = request.headers.get("accept-encoding", "")

        # Parse accepted encodings (handle gzip; q=1.0, br; q=0.9 format)
        supports_gzip = self._accepts_encoding(accept_encoding, "gzip")
        supports_brotli = self._accepts_encoding(accept_encoding, "br")

        # Skip compression if client doesn't support it
        if not supports_gzip and not supports_brotli:
            return await call_next(request)

        # Process request and get response
        response = await call_next(request)

        # Skip compression for streaming responses (they don't have .body attribute)
        if not hasattr(response, "body"):
            return response

        # Skip compression if response is already compressed
        content_encoding = response.headers.get("content-encoding", "")
        if content_encoding:
            return response

        # Skip compression for certain content types
        content_type = response.headers.get("content-type", "")
        if self._should_skip_compression(content_type):
            return response

        # Get response body
        try:
            response_body = response.body
        except AttributeError:
            # Response doesn't support body access (e.g., streaming)
            return response

        # Skip compression for small responses
        if len(response_body) < self.minimum_size:
            return response

        # Prefer brotli if supported and available
        if supports_brotli and self.brotli_available:
            return self._compress_brotli(response, response_body)

        # Fall back to gzip
        if supports_gzip:
            return self._compress_gzip(response, response_body)

        return response

    def _accepts_encoding(self, accept_encoding: str, encoding: str) -> bool:
        """Check if client accepts a specific encoding.

        Handles quality values (q) and wildcard accept-all.

        Args:
            accept_encoding: Value of Accept-Encoding header
            encoding: Encoding to check (e.g., "gzip", "br")

        Returns:
            True if encoding is accepted with q > 0

        """
        if not accept_encoding:
            return False

        accept_encoding_lower = accept_encoding.lower()

        # Check for wildcard
        if "*" in accept_encoding_lower:
            # Check if encoding is explicitly excluded with q=0
            parts = accept_encoding_lower.split(",")
            return all(not (encoding in part and ";q=0" in part.replace(" ", "")) for part in parts)

        # Check for specific encoding
        for part in accept_encoding_lower.split(","):
            part = part.strip()
            if part.startswith(encoding):
                # Check quality value
                if ";q=" in part:
                    try:
                        q_value = float(part.split(";q=")[1].strip())
                        return q_value > 0
                    except (ValueError, IndexError):
                        return True
                return True

        return False

    def _should_skip_compression(self, content_type: str) -> bool:
        """Determine if content type should skip compression.

        Skip compression for already compressed formats like images, videos,
        and compressed archives.

        Args:
            content_type: Response Content-Type header value

        Returns:
            True if compression should be skipped

        """
        # Content types that are already compressed
        skip_types = [
            "image/",  # Images are already compressed
            "video/",  # Videos are already compressed
            "audio/",  # Audio files are already compressed
            "application/zip",  # ZIP archives
            "application/x-gzip",  # GZIP archives
            "application/x-tar",  # TAR archives
            "application/x-compress",  # Compressed files
            "application/x-rar",  # RAR archives
            "application/x-7z",  # 7-Zip archives
            "application/pdf",  # PDF files (already compressed)
        ]

        content_type_lower = content_type.lower()
        return any(content_type_lower.startswith(skip_type) for skip_type in skip_types)

    def _compress_gzip(self, response: Response, body: bytes) -> Response:
        """Compress response using gzip.

        Args:
            response: Original response
            body: Response body to compress

        Returns:
            New response with compressed body and appropriate headers

        """
        # Create compressed buffer
        compressed_buffer = io.BytesIO()

        # Use zlib for better control over compression
        with gzip.GzipFile(
            fileobj=compressed_buffer,
            mode="wb",
            compresslevel=self.gzip_level,
        ) as gz_file:
            gz_file.write(body)

        compressed_data = compressed_buffer.getvalue()

        # Calculate compression ratio for logging
        original_size = len(body)
        compressed_size = len(compressed_data)
        ratio = (1 - compressed_size / original_size) * 100

        logger.debug(
            f"Gzip compression: {original_size} -> {compressed_size} bytes "
            f"({ratio:.1f}% reduction)",
        )

        # Create new response with compressed data
        return Response(
            content=compressed_data,
            status_code=response.status_code,
            headers={
                **dict(response.headers),
                "content-encoding": "gzip",
                "vary": "Accept-Encoding",
            },
            media_type=response.media_type,
        )

    def _compress_brotli(self, response: Response, body: bytes) -> Response:
        """Compress response using brotli.

        Args:
            response: Original response
            body: Response body to compress

        Returns:
            New response with compressed body and appropriate headers

        """
        import brotli

        # Compress data
        compressed_data = brotli.compress(
            body,
            quality=self.brotli_level,
        )

        # Calculate compression ratio for logging
        original_size = len(body)
        compressed_size = len(compressed_data)
        ratio = (1 - compressed_size / original_size) * 100

        logger.debug(
            f"Brotli compression: {original_size} -> {compressed_size} bytes "
            f"({ratio:.1f}% reduction)",
        )

        # Create new response with compressed data
        return Response(
            content=compressed_data,
            status_code=response.status_code,
            headers={
                **dict(response.headers),
                "content-encoding": "br",
                "vary": "Accept-Encoding",
            },
            media_type=response.media_type,
        )


# Factory function for easier configuration


def create_compression_middleware(
    minimum_size: int = 500,
    gzip_level: int = 6,
    brotli_level: int = 4,
) -> Callable[[ASGIApp], CompressionMiddleware]:
    """Factory function to create compression middleware with custom config.

    Allows easy configuration via environment variables or settings.

    Args:
        minimum_size: Minimum response size in bytes to compress
        gzip_level: Gzip compression level (1-9)
        brotli_level: Brotli compression level (0-11)

    Returns:
        Middleware factory function

    Example:
        from app.middleware.compression import create_compression_middleware

        app = FastAPI()
        app.add_middleware(create_compression_middleware(
            minimum_size=int(os.getenv("COMPRESSION_MIN_SIZE", "500")),
            gzip_level=int(os.getenv("COMPRESSION_LEVEL", "6")),
        ))

    """

    def middleware(app: ASGIApp) -> CompressionMiddleware:
        return CompressionMiddleware(
            app,
            minimum_size=minimum_size,
            gzip_level=gzip_level,
            brotli_level=brotli_level,
        )

    return middleware


__all__ = [
    "CompressionMiddleware",
    "create_compression_middleware",
]
