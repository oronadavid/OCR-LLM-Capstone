# ocr

# imports
import pytesseract
import cv2
import os
import argparse
import json
from PIL import Image

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return image

def extract_text(image_path):
    image = preprocess_image(image_path)
    return pytesseract.image_to_string(Image.fromarray(image), config='--psm 4 --oem 3')

def extract_text_with_bboxes(image_path):
    image = preprocess_image(image_path)
    data = pytesseract.image_to_data(Image.fromarray(image), output_type=pytesseract.Output.DICT, config='--psm 6 --oem 3')

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