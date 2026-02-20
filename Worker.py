# worker.py

import io
import fitz  # PyMuPDF
import httpx
import cv2
import numpy as np
from PIL import Image
import easyocr

from processing import (
    best_rotation_by_easyocr,
    extract_card,
    preprocess_for_compress_and_readability,
    resize_long_edge,
    quality_check,
    pil_to_png_bytes,
)

EXTRACTOR_URL = "https://file-extractordev.sidbi.in/extract"
EXTRACTOR_TIMEOUT = 60.0

# Global reader (one per worker process)
reader = None


def init_reader():
    """
    Ensures EasyOCR loads once per worker process.
    """
    global reader
    if reader is None:
        reader = easyocr.Reader(["en", "hi"], gpu=False)


def extractor_call(file_bytes: bytes, filename: str, content_type: str):
    """
    Synchronous extractor call (safe inside worker process).
    """
    with httpx.Client(timeout=EXTRACTOR_TIMEOUT, verify=False) as client:
        files = {"file": (filename, file_bytes, content_type)}
        resp = client.post(EXTRACTOR_URL, files=files)

    if resp.status_code < 200 or resp.status_code >= 300:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"ExtractorDev error {resp.status_code}: {detail}")

    return resp.json()


def process_document(raw: bytes, filename: str):
    """
    Entire heavy pipeline runs inside worker process.
    """

    init_reader()

    ext = (filename.split(".")[-1].lower() if filename else "")

    # ---- Load image or handle PDF ----
    if ext == "pdf":
        try:
            doc = fitz.open(stream=raw, filetype="pdf")
        except Exception:
            return {
                "status": "failure",
                "reason": "Decoding failed. Please retry again."
            }

        if len(doc) > 1:
            # Multipage PDF → forward directly
            try:
                extractor_json = extractor_call(
                    raw,
                    filename or "file.pdf",
                    "application/pdf"
                )
                return {"status": "success", "data": extractor_json}
            except Exception as e:
                return {
                    "status": "failure",
                    "reason": "extractor_failed",
                    "detail": str(e)
                }

        # Single-page PDF → render to image
        page = doc[0]
        zoom = 300 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        try:
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        except Exception:
            return {
                "status": "failure",
                "reason": "Decoding failed. Please retry again."
            }

    else:
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return {
                "status": "failure",
                "reason": "Decoding failed. Please retry again."
            }

    # ---- Rotation (Updated angles: 0, 90, -90) ----
    img, best_angle, best_stats, per_angle = best_rotation_by_easyocr(
        img,
        reader,
        angles=(0, 90, -90),
        max_long_edge=1100,
        conf_th=0.0
    )

    # ---- Card Detection ----
    card = extract_card(img, reader)
    if not card:
        return {
            "status": "failure",
            "reason": "Card not detected. Please make sure all edges of the cards are visible. "
                      "Note: Retake/re-upload in better quality—good lighting, sharp focus, no shadows/glare.",
            "rotation": {
                "best_angle": best_angle,
                "best_stats": best_stats,
                "per_angle": per_angle
            }
        }

    # ---- Preprocess ----
    card = preprocess_for_compress_and_readability(card)
    card = resize_long_edge(card)

    # ---- Quality Check ----
    cv_img = cv2.cvtColor(np.array(card), cv2.COLOR_GRAY2BGR)
    ok, reason = quality_check(cv_img, reader)

    if not ok:
        return {
            "status": "failure",
            "reason": reason,
            "rotation": {
                "best_angle": best_angle,
                "best_stats": best_stats,
                "per_angle": per_angle
            }
        }

    # ---- Extractor Call ----
    img_bytes = pil_to_png_bytes(card)

    try:
        extractor_json = extractor_call(
            img_bytes,
            "card.png",
            "image/png"
        )

        return {
            "status": "success",
            "rotation": {
                "best_angle": best_angle,
                "best_stats": best_stats,
                "per_angle": per_angle
            },
            "data": extractor_json
        }

    except Exception as e:
        return {
            "status": "failure",
            "reason": "extractor_failed",
            "detail": str(e),
            "rotation": {
                "best_angle": best_angle,
                "best_stats": best_stats,
                "per_angle": per_angle
            }
        }
