# --------------------------------------------------------------------------------------------------------
# Description: This script extracts text from bank statement images using Tesseract OCR.
# Author: Connor Bennett and David Orona
# Date: 2021-08-10`
# --------------------------------------------------------------------------------------------------------

# IMPORTSS
import pytesseract
import cv2
import os
import argparse
from PIL import Image

def preprocess_image(image_path):
    """Preprocesses the image for better OCR accuracy."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Scale up
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image

def extract_text(image_path):
    """Extracts text from a bank statement image using Tesseract OCR."""
    image = preprocess_image(image_path)
    text = pytesseract.image_to_string(Image.fromarray(image), config='--psm 4 --oem 3')
    print(f"\nExtracted text from {image_path}:\n{text}\n")  # Print extracted text for debugging
    return text

def process_folder(folder_path):
    """Processes all images in a folder and extracts text."""
    results = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {file_path}")
            extracted_text = extract_text(file_path)
            results[filename] = extracted_text
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from bank statement images in a folder.")
    parser.add_argument("folder", type=str, help="Path to the folder containing bank statement images.")
    args = parser.parse_args()
    
    extracted_results = process_folder(args.folder)
    for filename, text in extracted_results.items():
        print(f"\nExtracted text for {filename}:\n{text}")
