# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import asyncio

from executor import executor
from worker import process_document

app = FastAPI()


@app.post("/process")
async def process(file: UploadFile = File(...)):
    """
    Lightweight API layer.
    Offloads heavy OCR + processing to ProcessPool worker.
    """

    # Validate file extension (same logic as your original)
    ext = (file.filename.split(".")[-1].lower() if file.filename else "")
    raw = await file.read()

    if ext not in ["pdf", "jpg", "jpeg", "png"]:
        return JSONResponse({
            "status": "failure",
            "reason": "Invalid file type. Please upload either one of pdf, jpg, jpeg or png"
        })

    try:
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            executor,
            process_document,
            raw,
            file.filename or "file"
        )

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({
            "status": "failure",
            "reason": "Processing failed",
            "detail": str(e)
        })
