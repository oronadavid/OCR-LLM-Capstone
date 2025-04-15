# ocr.py

import pytesseract
import os
import argparse
import json
from PIL import Image
from preprocessing.ocr_preprocessing import preprocess_image

# Load preprocessing config from config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

# Load the config file
with open(CONFIG_PATH) as f:
    config = json.load(f)

# Load OCR preprocessing settings
pre_cfg = config.get("ocr_preprocessing", {})

"""
This module provides functions to perform OCR on images, including preprocessing steps
like grayscale conversion, upscaling, blurring, and adaptive thresholding.
It also includes functions to extract text and bounding box information from images.
"""
def extract_text(image_path):
    image = preprocess_image(
        image_path,
        use_grayscale=pre_cfg.get("use_grayscale", True),
        use_upscale=pre_cfg.get("use_upscale", True),
        use_blur=pre_cfg.get("use_blur", True),
        use_threshold=pre_cfg.get("use_threshold", True)
    )
    return pytesseract.image_to_string(Image.fromarray(image), config='--psm 4 --oem 3')

"""
Extracts text from an image using OCR.
Args:
    image_path (str): Path to the image file.
Returns:
    str: Extracted text.
"""
def extract_text_with_bboxes(image_path):
    image = preprocess_image(
        image_path,
        use_grayscale=pre_cfg.get("use_grayscale", True),
        use_upscale=pre_cfg.get("use_upscale", True),
        use_blur=pre_cfg.get("use_blur", True),
        use_threshold=pre_cfg.get("use_threshold", True)
    )
    data = pytesseract.image_to_data(
        Image.fromarray(image),
        output_type=pytesseract.Output.DICT,
        config='--psm 6 --oem 3'
    )

    extracted_data = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:
            extracted_data.append({
                "text": text,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "confidence": data["conf"][i]
            })
    return extracted_data

"""
Extracts text and bounding box information from an image using OCR.
Args:
    image_path (str): Path to the image file.
    output_dir (str): Directory to save the output files.
Returns:
    tuple: Extracted text and bounding box information.
"""
def process_image(image_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    raw_text = extract_text(image_path)
    bboxes = extract_text_with_bboxes(image_path)

    with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(os.path.join(output_dir, f"{filename}.json"), "w", encoding="utf-8") as f:
        json.dump(bboxes, f, indent=2)

    return raw_text, bboxes

"""
Processes a folder of images, applying OCR and saving the results.
Args:
    folder_path (str): Path to the folder containing images.
    output_dir (str): Directory to save the output files.
Returns:
    None
"""
def process_folder(folder_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, fname)
            print(f"[OCR] Processing {fname}")
            process_image(image_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on a folder of images.")
    parser.add_argument("folder", type=str, help="Folder of images to process.")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory for .txt and .json files")
    args = parser.parse_args()
    process_folder(args.folder, args.output)
