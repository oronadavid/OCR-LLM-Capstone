from ollama import chat
from ollama import ChatResponse

# ENTER THE OLLAMA NAMES OF THE MODELS TO TEST HERE v
models_to_use = ['gemma3:1b', 'llama3.2', 'deepseek-r1']

def run_llm_model(model, input_text, prompt):
    response: ChatResponse = chat(model=model, messages=[
      {
        'role': 'system',
        'content': prompt,
      },
      {
        'role': 'user',
        'content': input_text,
      },
    ])
    return(response['message']['content'])
