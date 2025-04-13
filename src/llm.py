# llm.py
from preprocessing.llm_preprocessing import clean_text
import json
from ollama import chat, ChatResponse

def analyze_bank_statement(text, bboxes, model="llama3.2"):
    response: ChatResponse = chat(model=model, messages=[
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
            'content': f"Extracted Text:\n{text}\n\nBounding Box Metadata:\n{json.dumps(bboxes)}"
        },
    ])
    return response['message']['content']

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM on OCR outputs.")
    parser.add_argument("model", type=str, help="Model to use")
    parser.add_argument("--input_dir", default="outputs", help="Directory with .txt and .json files")

    args = parser.parse_args()

    for fname in os.listdir(args.input_dir):
        if fname.endswith(".txt"):
            base = os.path.splitext(fname)[0]
            with open(os.path.join(args.input_dir, f"{base}.txt"), "r", encoding="utf-8") as f:
                raw_text = f.read()
            with open(os.path.join(args.input_dir, f"{base}.json"), "r", encoding="utf-8") as f:
                bboxes = json.load(f)
            print(f"\n[{args.model}] --- {base} ---")
            print(analyze_bank_statement(raw_text, bboxes, args.model))
