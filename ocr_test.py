import time
from io import BytesIO
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from pathlib import Path

import json


def main():
    config = {
        "output_format": "chunks",
        "filter_blank_pages": True,
        "extract_images": False,
        "disable_tqdm": True,

        "layout_batch_size": 48,
        "detection_batch_size": 48,
        "recognition_batch_size": 48,
        "table_rec_batch_size": 24,
        "equation_batch_size": 24,
        "ocr_error_batch_size": 24,

        "lowres_image_dpi": 120,
        "highres_image_dpi": 240,

        "disable_ocr": False,
        "disable_multiprocessing": False,
        "force_ocr": False,
        "disable_ocr_math": True,

        "max_concurrency": 14,
        "pdftext_workers": 14,
    }

    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )

    input_dir = Path("input")

    total_files = 0.0
    total_runtime = 0.0

    for pdf_file in input_dir.glob("*"):
        print(f"Processing {pdf_file}")

        # ---- START FILE TIMER ----
        file_start = time.perf_counter()

        try:
            with open(pdf_file, "rb") as f:
                pdf_bytes = f.read()

            pdf_stream = BytesIO(pdf_bytes)
            rendered = converter(pdf_stream)
            text, _, _ = text_from_rendered(rendered)

            output_file = Path("output") / f"{pdf_file.stem}.json"
            output_file.parent.mkdir(exist_ok=True)

            with open(output_file, "wb") as f:
                f.write(text.encode("utf-8"))

            # ---- END FILE TIMER ----
            runtime = time.perf_counter() - file_start
            print(f"✓ {pdf_file.name} finished in {runtime:.2f}s")

            total_files += 1
            total_runtime += runtime
            print(f"Avg runtime after {total_files:.0f} files is {total_runtime/total_files:.2f}s")

        except Exception as e:
            runtime = time.perf_counter() - file_start
            print(f"✗ {pdf_file.name} failed after {runtime:.2f}s")
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()