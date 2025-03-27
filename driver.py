# --------------------------------------------------------------------------------------------------------
# Description: Driver File for OCR & LLM
# Author: Connor Bennett and David Orona
# Date: 2021-08-10
# --------------------------------------------------------------------------------------------------------

import os
import argparse
import json
from ocr import preprocess_image, extract_text, extract_text_with_bboxes
from ollama import chat
from ollama import ChatResponse

# Models
available_models = ['gemma3:1b', 'llama3.2', 'deepseek-r1']

# Output folder
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_image(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Extract text and bounding boxes
    raw_text = extract_text(image_path)
    bboxes = extract_text_with_bboxes(image_path)

    # Save .txt
    text_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Save .json
    json_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bboxes, f, indent=2)

    return raw_text, bboxes

def run_ocr_and_llm(folder_path, model=None):
    models = [model] if model else available_models

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(folder_path, fname)
        print(f"Processing: {image_path}")

        # Process each image
        raw_text, bboxes = process_image(image_path)

        for model_name in models:
            print(f"\n{model_name.upper()} --- {fname} ---")
            try:
                response: ChatResponse = chat(model=model_name, messages=[
                    {
                        'role': 'system',
                        'content': (
                            "You're analyzing a bank statement. Use the extracted text and visual metadata (bounding boxes) "
                            "to return:\n"
                            "- All transactions (date, vendor, amount)\n"
                            "- Total debits/credits\n"
                            "- Starting and ending balances\n"
                            "- Overdraft or returned item fees\n"
                            "Respond in markdown or JSON."
                        ),
                    },
                    {
                        'role': 'user',
                        'content': f"Extracted Text:\n{raw_text}\n\nBounding Box Metadata:\n{json.dumps(bboxes)}"
                    },
                ])
                print(response['message']['content'])

            except Exception as e:
                print(f"⚠️  Failed on model {model_name} with {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on folder and LLM on each image.")
    parser.add_argument("folder", type=str, help="Folder of images to process.")
    parser.add_argument("model", nargs="?", default=None, help="(Optional) Single model to run.")

    args = parser.parse_args()
    run_ocr_and_llm(args.folder, args.model)
