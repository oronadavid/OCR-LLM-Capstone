# --------------------------------------------------------------------------------------------------------
# Description: This script extracts text from bank statement images using Tesseract OCR, including bounding boxes.
# Author: Connor Bennett and David Orona
# Date: 2021-08-10
# --------------------------------------------------------------------------------------------------------

# IMPORTS
import pytesseract
import cv2
import os
import argparse
import json
from PIL import Image

def preprocess_image(image_path):
    """Preprocesses the image for better OCR accuracy."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Scale up
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image

def extract_text_with_bboxes(image_path):
    """Extracts text from a bank statement image using Tesseract OCR with bounding boxes."""
    image = preprocess_image(image_path)
    
    # Use Tesseract to get OCR results with bounding boxes
    data = pytesseract.image_to_data(Image.fromarray(image), output_type=pytesseract.Output.DICT, config='--psm 6 --oem 3')
    
    extracted_data = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:  # Ignore empty results
            extracted_data.append({
                "text": text,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "confidence": data["conf"][i]
            })
    
    return extracted_data

def extract_text(image_path):
    """Extracts plain text from a bank statement image using Tesseract OCR."""
    image = preprocess_image(image_path)
    text = pytesseract.image_to_string(Image.fromarray(image), config='--psm 4 --oem 3')
    return text


def process_folder(folder_path):
    """Processes all images in a folder and extracts text with bounding boxes."""
    results = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {file_path}")
            extracted_data = extract_text_with_bboxes(file_path)
            results[filename] = extracted_data

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from bank statement images with bounding boxes.")
    parser.add_argument("folder", type=str, help="Path to the folder containing bank statement images.")
    parser.add_argument("--output", type=str, default="output.json", help="Output JSON file to store results.")
    args = parser.parse_args()
    
    extracted_results = process_folder(args.folder)

    # Save results to JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(extracted_results, f, indent=4)

    print(f"\nResults saved to {args.output}")


"""                                                                               TempiateLAB

October 01, 2019, through November 30, 2019
B BY. Account Number: 254 100541522695
B
"""