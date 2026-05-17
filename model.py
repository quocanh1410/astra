
import ollama

SLM_MODEL = "qwen2:0.5b"  # yếu hơn → SLM
LLM_MODEL = "phi3:mini"   # mạnh hơn → LLM

def call_model(model_name, messages, options):
    res = ollama.chat(
        model=model_name,
        messages=messages,
        options=options,
        keep_alive=0  # unload sau khi xong, tiết kiệm RAM
    )
    return res['message']['content']

def call_slm(prompt: str) -> str:
    return call_model(
        SLM_MODEL,
        messages=[
            {"role": "system", "content": "Answer briefly and directly and exactly."},
            {"role": "user",   "content": prompt}
        ],
        options={"temperature": 0.1, "num_predict": 256}
    )

def call_llm(prompt: str) -> str:
    return call_model(
        LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "Think carefully step by step. "
                "Explain your reasoning before answering."
            )},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.7, "num_predict": 256}
    )