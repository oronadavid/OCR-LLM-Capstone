import re
import json

def clean_text(text):
    lines = text.splitlines()
    return "\n".join([line.strip() for line in lines if line.strip()])

def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text)

def remove_headers_footers(text):
    return re.sub(r'Page \d+ of \d+|CONFIDENTIAL', '', text, flags=re.IGNORECASE)

def standardize_dates(text):
    pattern = re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b')
    def fix(match):
        month, day, year = match.groups()
        year = f"20{year}" if len(year) == 2 else year
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return pattern.sub(fix, text)

def standardize_amounts(text):
    return re.sub(r'\$?(-?\d[\d,]*\.?\d{0,2})', lambda m: str(float(m.group(1).replace(',', ''))), text)

def extract_transaction_lines(text):
    pattern = re.compile(r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b.+?\$?\s*-?\d[\d,]*\.?\d{0,2}')
    return "\n".join([line for line in text.splitlines() if pattern.search(line)])

def preprocess_text(
    text,
    clean=True,
    normalize=True,
    remove_headers=True,
    standardize_dates_flag=True,
    standardize_amounts_flag=True,
    extract_lines=True
):
    if clean:
        text = clean_text(text)
    if normalize:
        text = normalize_whitespace(text)
    if remove_headers:
        text = remove_headers_footers(text)
    if standardize_dates_flag:
        text = standardize_dates(text)
    if standardize_amounts_flag:
        text = standardize_amounts(text)
    if extract_lines:
        text = extract_transaction_lines(text)
    return text

def format_with_spacing(json_path):
    data = {}

    with open(json_path) as f:
        data = json.load(f)

    for image in data:
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
            output_text += text_line + "\n"

        with open(f"output/spaced_formatted_{image}.txt", "w") as f:
            f.write(output_text)
