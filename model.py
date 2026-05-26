
# import ollama

# SLM_MODEL = "qwen2:0.5b"  # yếu hơn → SLM
# LLM_MODEL = "phi3:mini"   # mạnh hơn → LLM

# def call_model(model_name, messages, options):
#     res = ollama.chat(
#         model=model_name,
#         messages=messages,
#         options=options,
#         keep_alive=0  # unload sau khi xong, tiết kiệm RAM
#     )
#     return res['message']['content']

# def call_slm(prompt: str) -> str:
#     return call_model(
#         SLM_MODEL,
#         messages=[
#             {"role": "system", "content": "Answer briefly and directly and exactly."},
#             {"role": "user",   "content": prompt}
#         ],
#         options={"temperature": 0.1, "num_predict": 256}
#     )

# def call_llm(prompt: str) -> str:
#     return call_model(
#         LLM_MODEL,
#         messages=[
#             {"role": "system", "content": (
#                 "Think carefully step by step. "
#                 "Explain your reasoning before answering."
#             )},
#             {"role": "user", "content": prompt}
#         ],
#         options={"temperature": 0.7, "num_predict": 256}
#     )



# import ollama

# SLM_MODEL = "qwen2:0.5b"
# LLM_MODEL = "phi3:mini"

# def call_model(model_name, messages, options):
#     res = ollama.chat(
#         model=model_name,
#         messages=messages,
#         options=options,
#         keep_alive=0
#     )
#     return res['message']['content']

# FEW_SHOT = [
#     {"role": "user",      "content": "What is the capital of France?"},
#     {"role": "assistant", "content": "Paris"},
#     {"role": "user",      "content": "Who wrote Hamlet?"},
#     {"role": "assistant", "content": "Shakespeare"},
#     {"role": "user",      "content": "What element has symbol Au?"},
#     {"role": "assistant", "content": "Gold"},
# ]

# def call_slm(prompt: str) -> str:
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a math solver. "
#                 "Reply with ONLY the final answer. "
#                 "No words, no units, no explanation."
#             )
#         }
#     ] + FEW_SHOT + [
#         {"role": "user", "content": prompt}
#     ]
#     return call_model(
#         SLM_MODEL,
#         messages=messages,
#         options={"temperature": 0.0, "num_predict": 32}
#     )

# def call_llm(prompt: str) -> str:
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a math solver. "
#                 "Reply with ONLY the final answer. "
#                 "No words, no units, no explanation."
#             )
#         }
#     ] + FEW_SHOT + [
#         {"role": "user", "content": prompt}
#     ]
#     return call_model(
#         LLM_MODEL,
#         messages=messages,
#         options={"temperature": 0.0, "num_predict": 32}
#     )




# model.py
# model.py
# model.py
import os
import re
import ollama
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
SLM_MODEL   = "qwen2:0.5b"               # local Ollama
LLM_MODEL   = "llama-3.3-70b-versatile"  # Groq

SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "At the end, write 'Answer: X' where X is the final number only."
)

def call_slm(prompt: str) -> str:
    """SLM yếu chạy local — dùng khi URC route sang SLM."""
    res = ollama.chat(
        model=SLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        options={"temperature": 0.0, "num_predict": 256},
        keep_alive=0
    )
    return res['message']['content']

def call_llm(prompt: str) -> str:
    """LLM mạnh trên Groq — dùng khi URC route sang LLM."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

def extract_number(text: str) -> str:
    match = re.search(r'Answer:\s*(\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    numbers = re.findall(r'\d+\.?\d*', text)
    return numbers[-1] if numbers else ""