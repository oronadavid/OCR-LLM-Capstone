import ocr
import llm
import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from bank statement images with bounding boxes.")
    parser.add_argument("folder", type=str, help="Path to the folder containing bank statement images.")
    parser.add_argument("--output", type=str, default="output.json", help="Output JSON file to store results.")
    parser.add_argument("--model", type=str, default="llama3.2", help="LLM model to use for processing")
    args = parser.parse_args()
    
    extracted_results = ocr.process_folder(args.folder)

    # Save results to JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(extracted_results, f, indent=4)

    print(f"\nResults saved to {args.output}")

    prompt = 'Given the following bank statement, give me a list of all of the withdrawls and deposits with their dollar amounts.'

    folder_path = "output"
    ocr.format_with_spacing(args.output)
    print("BEGIN PROCESSING----------------------")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print("\n\n\n\n\n")
        print(f"Processing: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            ocr_text = f.read()
            #print(ocr_text)
            #print(prompt)
            llm_output = llm.run_llm_model(args.model, ocr_text, prompt)
            print(llm_output)
