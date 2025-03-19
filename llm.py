from ollama import chat
from ollama import ChatResponse

# ENTER THE OLLAMA NAMES OF THE MODELS TO TEST HERE v
models_to_use = ['gemma3:1b', 'llama3.2', 'deepseek-r1']

ocr_input = '''
Ww ABC BANK Statement Ending 09/21/2021 Page 2 of 2

Look over all transactions

ABC seumnbens CHEKING XXXXXXXX1284 (continued)
Primary Chekin Paycheck

     

  
   
  
    

Account Activity

Post Date Debits Credits Balance
09/10/2021 Signature POS Debit 09/08 GIANT FOOD I $78.22 6,806.09
09/10/2021 LA FITNESS $15.38 6,790.71

09/11/2021 FT&T MOBILITY ONLINE PMT
09/14/2021 DEPOSIT
09/14/2021 DIRECT DEP

$100.30 6,690.41

7,190.89

 

 

 

 

 

 

09/14/2021 Signature POS Debit 09/13 MD GIANT FOOD $15.40 9,751 83
09/17/2021_ ATM Withdrawal_09/15 WV INWOOD. $350.00 9,401.83
09/17/2021 Signature POS Debit 09/16 MD GIANT FOOD $12.48 9,389.35
09/17/2021 Signature POS Debit 09/15 MD GIANT FOOD $35.80 9,353.55
09/18/2021 THE HOME DEPOT ONLINE PMT POS $8.35 9,345.20
09/18/2021 QORETIRE 0503R3030 $528.12 8,817.08
09/20/2021 323LA@71557195 $489.57 8,327.51
09/21/2021 Ending Balance 8,327.51
Daily Balances

Date Amount Date Amount _Date Amount
09/04/2021 19,120.43 09/10/2021 6,790.71 09/18/2021 8,817.08
09/05/2021 7,017.64 09/11/2021 6,690.41 09/20/2021 8,327.51
09/06/2021 6,562.32 09/14/2021 9,751,83

09/07/2021 7,039.31 09/17/2021 9,353.55

Overdraft and Returned Item Fees Watch for unexpected fees

P Total for this for this period / Total year-to-date
Total Overdraft Fees 7 0 8000] 00

 

Total Returned Tem Fees PT
'''

for model in models_to_use:
    print(f"{model.upper()}-------------------------------------------")
    response: ChatResponse = chat(model=model, messages=[
      {
        'role': 'system',
        'content': 'Given the following bank statement, give me a list of all of the transations with their dollar amounts. Give me all other listed balance and expense information as well.',
      },
      {
        'role': 'user',
        'content': ocr_input,
      },
    ])
    print(response['message']['content'])
# or access fields directly from the response object
#print(response.message.content)
