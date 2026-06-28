"""Encode local image files for multimodal provider payloads."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


class ImagePayloadError(ValueError):
    """Raised when an image file cannot be converted into a provider payload."""


def image_data_url(image_path: Path) -> str:
    """Return a data URL for an image file accepted by chat-completion payloads."""
    mime_type, _encoding = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ImagePayloadError(f"{image_path}: unsupported image type")

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"
