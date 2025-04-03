# driver.py
import sys
import os

# Make sure 'src/' is in the import path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ocr import process_image
from llm import analyze_bank_statement
import argparse
import json

# run using python src/driver.py images/test_images llama3.2:latest
#                                   image location    model name

def run(folder_path, model):
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(folder_path, fname)
        print(f"[DRIVER] Processing: {image_path}")
        raw_text, bboxes = process_image(image_path)

        print(f"\n[{model}] --- {fname} ---")
        response = analyze_bank_statement(raw_text, bboxes, model)
        print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for OCR + LLM.")
    parser.add_argument("folder", type=str, help="Folder of images")
    parser.add_argument("model", type=str, help="Model to use (e.g., llama3.2)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise FileNotFoundError(f"Input folder '{args.folder}' does not exist.")

    run(args.folder, args.model)
