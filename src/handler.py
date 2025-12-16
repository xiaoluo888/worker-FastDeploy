import runpod
import base64
import tempfile
import os
from paddlex import create_pipeline
from runpod import RunPodLogger

log = RunPodLogger()

log.info("Booting PaddleOCR-VL pipeline...")
pipeline = create_pipeline(
    "PaddleOCR-VL",
    config="/home/paddleocr/pipeline_config_fastdeploy.yaml"
)
log.info("Pipeline ready.")

def handler(event):
    try:
        log.info("Received OCR request")
        input_data = event.get("input", {})

        file_base64 = input_data.get("file_base64")
        file_type = input_data.get("file_type", "image")  # image | pdf

        if not file_base64:
            log.error("file_base64 missing in input")
            return {"status": "error", "error": "file_base64 missing"}

        # Decide suffix
        if file_type.lower() == "pdf":
            suffix = ".pdf"
        else:
            suffix = ".png"

        log.debug(f"Processing file of type: {file_type}")

        file_bytes = base64.b64decode(file_base64)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(file_bytes)
            file_path = f.name

        try:
            log.info("Starting OCR prediction...")
            results = list(pipeline.predict(file_path))
            log.info(f"OCR prediction completed, found {len(results)} pages")

            pages = []
            for i, r in enumerate(results):
                pages.append({
                    "page_number": i + 1,
                    "result": (
                        r.json if hasattr(r, "json")
                        else r.to_dict() if hasattr(r, "to_dict")
                        else r
                    )
                })

            log.info(f"Processed {len(pages)} pages successfully")
            return {
                "status": "success",
                "num_pages": len(pages),
                "pages": pages
            }

        finally:
            os.remove(file_path)
            log.debug("Temporary file removed")

    except Exception as e:
        log.error(f"Error processing OCR request: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

runpod.serverless.start({"handler": handler})
