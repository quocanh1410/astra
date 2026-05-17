
import ollama


MODEL = "mistral"


def call_slm(prompt):

    res = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer briefly and directly."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1,
            "num_predict": 64
        }
    )

    return res['message']['content']


def call_llm(prompt):

    res = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Think carefully step by step. "
                    "Explain your reasoning before answering. "
                    "Double check your answer."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.7,
            "num_predict": 256
        }
    )

    return res['message']['content']


q = "Why do airplanes fly?"


print("\n=== SLM ===\n")
print(call_slm(q))

print("\n=== LLM ===\n")
print(call_llm(q))

