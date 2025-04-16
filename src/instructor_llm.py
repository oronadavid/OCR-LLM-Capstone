from typing import List, Literal
import instructor
from openai import OpenAI
from pydantic import BaseModel
from datetime import date

import ollama

class BankStatementEntry(BaseModel):
    transaction_date: date | None
    description: str | None
    amount: float | None
    transaction_type: Literal['deposit', 'withdrawal', None]

class BankStatement(BaseModel):
    transactions: List[BankStatementEntry]

def pull_ollama_model(model: str):
    """
    Pull a model from ollama if it is not already downloaded
    """
    for downloaded_model in ollama.list()["models"]:
        if downloaded_model == model:
            return
    
    print(f"Downloading {model} model...")
    ollama.pull(model)

def extract_json_data_using_ollama_llm(prompt: str, text_data: str, ollama_model: str) -> str:
    """
    Pass prompt and data into an ollama LLM using instructor
    """
    pull_ollama_model(ollama_model)

    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        ),
        mode=instructor.Mode.JSON
    )

    resp = client.chat.completions.create(
        model=ollama_model,
        messages=[
            {
                'role': 'system',
                'content': prompt
            },
            {
                'role': 'user',
                'content': text_data
            },
        ],
        response_model=BankStatement,
        max_retries=3
    )

    return resp.model_dump_json(indent=4)

if __name__ == "__main__":
    prompt = "Extract all transactions from the following statement. Each transaction must be returned as a JSON object with the fields: transaction_date (YYYY-MM-DD), description, amount, and transaction_type ('deposit' or 'withdrawal'). All of these must be returned as a list of JSON objects under a key called 'transactions'."

    bank_statement = """
    ## Look over all transactions

    ## ABC RELATIONSHIP CHEKING XXXXXXXXI284 (conlinued)

    | Primary Cheking   | Primary Cheking                                      |         | Paycheck   | Paycheck   |
    |-------------------|------------------------------------------------------|---------|------------|------------|
    | Account Activity  |                                                      | Dehits  | Credits    | Balance    |
    |                   | 09/10/2021   Signalure POS Debil 09/08 GIANT FOOD I  | 578.22  |            | 6,806,09   |
    | 09/10/2021        | LA FITNESS                                           | 515.38  |            | 6,790.71   |
    |                   | 09/11/2021 FT&T MOBILITY ONLINE PMT                  |         |            | 6,690,41   |
    | (9/14/2021        | DEPOSIT                                              |         |            | 7,190,89   |
    | 09/14/2021        | DIRECT DEP                                           |         | 52,576.34  | 9,767.23   |
    |                   | 09/14/2021  Signaturc POS Dcbit 09/13 MD GIANT FOOD  | 515,40  |            | 9,751,83   |
    | 09/17/2021        | ATM Withdrawal   09/15 WV INWOOD                     |         |            | 9,401,83   |
    |                   | 09/17/2021   Signalure POS Debil 09/16 MD GIANT FOOD | 512.48  |            | 9,389.35   |
    |                   | 09/17/2021  Signature POS Dcbit 09/15 MD GIANT FOOD  | 535.80  |            | 9,353.55   |
    | 09/18/2021        | THE HOME DEPOT ONLINE PMT POS                        |         |            | 9,345.20   |
    | 09/18/2021        |                                                      | 5528.12 |            | 8,817.08   |
    | (9/20/2021        | 323LA@71557195                                       | 5489.57 |            | 8,327.51   |
    | 09/21/2021        | Ending Balance                                       |         |            | 8,327.51   |

    ## Balances Daily

    | Date       | Amount   | Date       | Amnunt   | Date       | Amount   |
    |------------|----------|------------|----------|------------|----------|
    | 09/04'2021 |          | 09/10/2021 | 6,790.71 | 09/18/2021 | 8,817,08 |
    | 09/05/2021 | 7,017.64 | 09/11/2021 | 6,690.41 | 09/20/2021 | 8,32751  |
    | 09/06/2021 | 6,562.32 | 09/14/2021 | 9,751,83 |            |          |
    |            | 7,039.31 | U9/17/2021 | 9,353.55 |            |          |

    ## Overdraft and Returned Item Fees

    ## Watch for unexpected fees

    |                         | Total for this period   | Total year-to-date   |
    |-------------------------|-------------------------|----------------------|
    | Total Overdraft Fees    |                         |                      |
    | Total Returned Tem Fees |                         |                      |
                 |                      |
    """

    print(extract_json_data_using_ollama_llm(prompt=prompt, text_data=bank_statement, ollama_model="llama3.2"))
