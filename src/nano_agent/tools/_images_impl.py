from __future__ import annotations

from ._common import *

def supports_view_image_original_detail(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized == "gpt-5.3-codex"


def supports_image_inputs(provider: str, model: str) -> bool:
    normalized_provider = provider.strip().lower()
    if normalized_provider == "copilot":
        try:
            entry = _model_catalog_entry(model, provider=provider)
        except Exception:
            entry = None
        capabilities = (
            entry.get("capabilities")
            if isinstance(entry, dict) and isinstance(entry.get("capabilities"), dict)
            else {}
        )
        inputs = (
            capabilities.get("input_modalities")
            if isinstance(capabilities, dict)
            else None
        )
        if isinstance(inputs, list) and inputs:
            return any(
                isinstance(item, str) and item.strip().lower() == "image"
                for item in inputs
            )
    return True


def _guess_supported_image_mime(path: Path, image: Image.Image) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        return guessed
    image_format = (image.format or "").upper()
    if image_format == "PNG":
        return "image/png"
    if image_format in {"JPEG", "JPG"}:
        return "image/jpeg"
    if image_format == "WEBP":
        return "image/webp"
    if image_format == "GIF":
        return "image/gif"
    return "image/png"


def _image_output_format_for_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return "JPEG"
    if mime_type == "image/webp":
        return "WEBP"
    if mime_type == "image/gif":
        return "GIF"
    return "PNG"


def _image_data_url(mime_type: str, data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _data_url_to_anthropic_source(data_url: str) -> dict[str, Any]:
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("image_url must be a data URL")
    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("image_url data URL must be base64-encoded")
    mime_type = header[5:].split(";", 1)[0].strip().lower()
    if mime_type not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
        raise ValueError(f"unsupported image mime type for Anthropic: {mime_type}")
    return {"type": "base64", "media_type": mime_type, "data": encoded}


def make_view_image_result(
    path: Path,
    *,
    detail: str | None,
    model: str,
    provider: str,
) -> dict[str, Any]:
    if not supports_image_inputs(provider, model):
        raise ValueError(
            "view_image is not allowed because the current model may not support image inputs"
        )

    normalized_detail = detail.strip() if isinstance(detail, str) else None
    if normalized_detail == "":
        normalized_detail = None
    if normalized_detail is not None and normalized_detail != "original":
        raise ValueError(
            "view_image.detail only supports `original`; omit `detail` for default resized behavior, got "
            + repr(normalized_detail)
        )
    if normalized_detail == "original" and not supports_view_image_original_detail(
        model
    ):
        raise ValueError(
            "view_image.detail only supports `original` on compatible models; omit `detail` for default resized behavior"
        )

    if not path.exists():
        raise ValueError(f"unable to locate image at `{path}`")
    if not path.is_file():
        raise ValueError(f"image path `{path}` is not a file")

    file_bytes = path.read_bytes()
    try:
        with Image.open(path) as source_image:
            source_image.load()
            mime_type = _guess_supported_image_mime(path, source_image)
            original_width, original_height = source_image.size
            use_original = normalized_detail == "original"
            output_image = source_image
            output_width, output_height = original_width, original_height
            if not use_original and (
                original_width > VIEW_IMAGE_MAX_WIDTH
                or original_height > VIEW_IMAGE_MAX_HEIGHT
            ):
                scale = min(
                    VIEW_IMAGE_MAX_WIDTH / original_width,
                    VIEW_IMAGE_MAX_HEIGHT / original_height,
                )
                output_width = max(1, int(math.floor(original_width * scale)))
                output_height = max(1, int(math.floor(original_height * scale)))
                output_image = source_image.resize(
                    (output_width, output_height), resample=Image.Resampling.BILINEAR
                )
            elif use_original or mime_type in {"image/png", "image/jpeg", "image/webp"}:
                return {
                    "success": True,
                    "path": str(path),
                    "image_url": _image_data_url(mime_type, file_bytes),
                    "detail": "original" if use_original else None,
                    "mime_type": mime_type,
                    "width": original_width,
                    "height": original_height,
                }

            from io import BytesIO

            buffer = BytesIO()
            output_format = _image_output_format_for_mime(mime_type)
            save_kwargs: dict[str, Any] = {}
            if output_format == "JPEG":
                save_kwargs["quality"] = 85
                if output_image.mode not in {"RGB", "L"}:
                    output_image = output_image.convert("RGB")
            output_image.save(buffer, format=output_format, **save_kwargs)
            data = buffer.getvalue()
            return {
                "success": True,
                "path": str(path),
                "image_url": _image_data_url(mime_type, data),
                "detail": "original" if use_original else None,
                "mime_type": mime_type,
                "width": output_width,
                "height": output_height,
            }
    except ValueError:
        raise
    except Exception as exc:
        mime_type, _ = mimetypes.guess_type(str(path))
        mime_label = mime_type or "unknown"
        raise ValueError(
            f"unable to process image at `{path}`: unsupported image `{mime_label}`"
        ) from exc
