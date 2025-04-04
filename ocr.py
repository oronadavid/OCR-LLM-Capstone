# --------------------------------------------------------------------------------------------------------
# Description: This script extracts text from bank statement images using Tesseract OCR, including bounding boxes.
# Author: Connor Bennett and David Orona
# Date: 2025-03-27
# --------------------------------------------------------------------------------------------------------

# TODO: Handle text that is not black on white
# TODO: Ex: In test4.png (First Platypus Bank), the black bar is not read

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

def format_with_spacing(json_path):
    data = {}

    with open(json_path) as f:
        data = json.load(f)

    for image in data:
        #print("\n\n\n")
        #print(image)

        # Determine rows
        text_row_dictionary = {}
        current_text = ""
        current_line_top = 0
        previous_item = None
        for text_item in data[image]:
            if previous_item is not None:
                # The item is on a new line
                if abs(previous_item['top'] - text_item['top']) > 25:

                    # Create a new row since this is a new column
                    text_row_dictionary[text_item['top']] = []

                    # Append the text being held
                    text_row_dictionary[current_line_top].append(current_text)

                    # Reset the text being held as this is a new text block on a new line
                    current_text = ""

                    # Save the current top number as it will be the dict index
                    current_line_top = text_item['top']

                # The item is not next to the previous item
                elif abs(previous_item['left'] + previous_item['width'] - text_item['left']) > 18:

                    # Append the text being held
                    text_row_dictionary[current_line_top].append(current_text)

                    # Reset the text being held as this is a new text block on the same line
                    current_text = ""

            else:
                # Set the initial current top number as the first element's top
                current_line_top = text_item['top']
                # Create a row for the initial column
                text_row_dictionary[text_item['top']] = []

            # Append the current text, either a new column has started, we are adding to the current text block, or we are starting a new text block
            current_text += text_item['text'] + " "
            previous_item = text_item

        output_text = ""
        for item in text_row_dictionary:
            text_line = ""
            for text_block in text_row_dictionary[item]:
                text_line += f"{text_block} |  "
            text_line = text_line[:-3]
            #print(text_line)
            output_text += text_line + "\n"

        with open(f"output/spaced_formatted_{image}.txt", "w") as f:
            f.write(output_text)


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
