import os
import json
from preprocessing.llm_preprocessing import preprocess_text
from ollama import chat, ChatResponse

# Load LLM preprocessing settings from config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(CONFIG_PATH) as f:
    config = json.load(f)

llm_cfg = config.get("llm_preprocessing", {})


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
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM on OCR outputs.")
    parser.add_argument("model", type=str, help="Model to use")
    parser.add_argument("--input_dir", default="outputs", help="Directory with .txt and .json files")

    args = parser.parse_args()

    for fname in os.listdir(args.input_dir):
        if fname.endswith(".txt"):
            base = os.path.splitext(fname)[0]

            # Load raw OCR text and bounding boxes
            with open(os.path.join(args.input_dir, f"{base}.txt"), "r", encoding="utf-8") as f:
                raw_text = f.read()
            with open(os.path.join(args.input_dir, f"{base}.json"), "r", encoding="utf-8") as f:
                bboxes = json.load(f)

            # 🧼 Apply preprocessing steps to the OCR text based on config
            preprocessed_text = preprocess_text(
                raw_text,
                clean=llm_cfg.get("clean_text", True),
                normalize=llm_cfg.get("normalize_whitespace", True),
                remove_headers=llm_cfg.get("remove_headers_footers", True),
                standardize_dates_flag=llm_cfg.get("standardize_dates", True),
                standardize_amounts_flag=llm_cfg.get("standardize_amounts", True),
                extract_lines=llm_cfg.get("extract_transaction_lines", True)
            )

            print(f"\n[{args.model}] --- {base} ---")
            print(analyze_bank_statement(preprocessed_text, bboxes, args.model))
