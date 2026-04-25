"""Codex-style view_image tool for local image files."""

from __future__ import annotations

import mimetypes
import stat as stat_mod
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar

from ..data_structures import ImageContent, TextContent
from ..execution_context import ExecutionContext
from .base import (
    SUPPORTED_IMAGE_TYPES,
    Desc,
    Tool,
    TruncationConfig,
    load_image_as_content,
)


@dataclass
class ViewImageInput:
    """Input for ViewImageTool."""

    path: Annotated[str, Desc("Local filesystem path to an image file")]
    detail: Annotated[
        str,
        Desc(
            "Optional detail override. Accepted values: omit or `original`; "
            "both currently return the image at its original resolution "
            "(nano-agent does not resize)."
        ),
    ] = ""


@dataclass
class ViewImageTool(Tool):
    """View a local image from the filesystem."""

    name: str = "view_image"
    description: str = (
        "View a local image from the filesystem (only use if given a full filepath "
        "by the user, and the image isn't already attached to the thread context "
        "within <image ...> tags)."
    )

    _truncation_config: ClassVar[TruncationConfig] = TruncationConfig(enabled=False)

    async def __call__(
        self,
        input: ViewImageInput,
        execution_context: ExecutionContext | None = None,
    ) -> ImageContent | TextContent:
        """Load an image file as an image content block."""
        if input.detail not in {"", "original"}:
            return TextContent(
                text=(
                    "Error: view_image.detail only supports `original`; omit "
                    f"`detail` for default resized behavior, got `{input.detail}`"
                )
            )

        path = Path(input.path).expanduser()
        try:
            st = path.stat()
        except FileNotFoundError:
            return TextContent(text=f"Error: unable to locate image at `{path}`")
        except PermissionError:
            return TextContent(text=f"Error: permission denied reading image `{path}`")
        except OSError as exc:
            return TextContent(text=f"Error: unable to read image at `{path}`: {exc}")

        if not stat_mod.S_ISREG(st.st_mode):
            return TextContent(text=f"Error: image path `{path}` is not a file")

        guessed_type, _ = mimetypes.guess_type(path.name)
        if guessed_type not in SUPPORTED_IMAGE_TYPES:
            return TextContent(
                text=(
                    f"Error: unsupported image type for `{path}`. Supported types: "
                    "PNG, JPEG, GIF, WebP."
                )
            )

        return load_image_as_content(path, guessed_type, file_size=st.st_size)
