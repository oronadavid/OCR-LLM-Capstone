import re

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
