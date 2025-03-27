from ollama import chat
from ollama import ChatResponse

# ENTER THE OLLAMA NAMES OF THE MODELS TO TEST HERE v
models_to_use = ['llama3.2']

ocr_input = '''
1 =CHOICE                                                                               duly 1, 2018 through July 31, 2018
                                                                                        Primary Account, 00000958581485
URNGrtin Choice Bank ,
Weet Vigra.
Courky Proada, WIV 70826-0100                                                           CUSTOMER SERVICE INFORMATION
                                                                                        website: weer. choleebank.com
                                                                                        “Serioe Center: “1-800-555-6035
                                                                                        iapeleneventanety bal Tae ld ay iNet iniematonal Calls: eee, a —a
Company Name                                                                            Contact ua by phono for questions, on this 5
Company Adcroas                                                                         ‘statement, change Information, and general —_—_—_
State, Zip.                                                                             inquiries, 24 hours a cay, 7 cays a woek —_—,
—=
Account Summary 
——*==,=
Opening Balance                                           $5,234.09 =
Withdravaals                                              $2,395.67 —
Deposits                                                  $2,872.45
Closing Balanco on Apr 18,2010                            $9,710.87

Your Transaction Detalls
Dato              Detalls              Withdrawals              Deposits:              Balance
Ape 8              Opening Balance.                                                    5,234.09
Apes               Insurance                                    272.45                 5,506.54
Ape 10             A™                    200.00                                        5,306.54
Ape 12             Intesnet Transfer     250,00                                        5,556.54
Ape 12             Payroll                                      2100.00                ‘7,656.54
Ape 13             Bill payment           135.07                                       ‘7,521.47
Apr 14             Direct debit                                 200.00                 7,821.47
Apr 14             Deposit               250.00                                        ‘7,567.87
Ape 15             Bill payment          525.72                                        7,042.15"
Ape 17             Bill payment          327.63                                        6,714,52
Ape?               Bill payment          729.98                                        5,984.56
Ape 18             Insurance             272.45                                        5,508.54
Ape 18-            AT™                   200.00                                        5,306.54
Apr 18             Intemet Transfer                             250,00                 5,556.54
Ape 18-            Payroll                                      2100,00                "7,656.54
Ape 18             Bill payment          135.07                                        7,521.47
Ape 19             Oirect debit          200.00                                        ‘7,321.47
Ape 19             Deposit                                      250.00                 "7.567.867
Apr 19             Bill payment          525.72                                        7,042.15
Ape 20             Sil payment           327.63                                        6,714.52,
Ape 20             Bil payment           729.96                                        5,984.56
Apr 20             Deposit                                      250,00                 "7,567.87.
Ape 20             Bill payment          525.72                                        “7,042.15
Ape 20             Bill payment          327.63                                        '6,714,52
Ape 23             Bill payment          729.96                                        5,984.56

Closing Balance $9,710.87


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
