from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, ImageFormatOption
import os
from huggingface_hub import snapshot_download

source = Path("/home/david/Projects/school/capstone/ocr-llm-setup/images/test_images/test1.png")

def load_rapid_ocr_model(det_model: str, rec_model: str, cls_model: str) -> DocumentConverter:
    print("Downloading RapidOCR models")
    download_path = snapshot_download(repo_id="SWHL/RapidOCR")

    det_model_path = os.path.join(
        download_path, "PP-OCRv4", "ch_PP-OCRv4_det_server_infer.onnx"
    )
    rec_model_path = os.path.join(
        download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx"
    )

    ocr_options = RapidOcrOptions(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path
    )

    pipeline_options = PdfPipelineOptions(
        ocr_options=ocr_options
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_options=pipeline_options
            ),
        }
    )

    return doc_converter

def image_to_text(document_converter: DocumentConverter):
    conv_results = document_converter.convert(source)
    print(conv_results.document.export_to_markdown())


if __name__ == "__main__":
    document_converter = load_rapid_ocr_model("","","")
    image_to_text(document_converter)
