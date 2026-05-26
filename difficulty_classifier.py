import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_difficulty_score(request: str) -> float:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "Count the number of calculation steps needed to solve this math problem.\n"
                    "Reply with ONLY a single integer.\n"
                    "Examples:\n"
                    "- 'Tom has 5 apples, buys 3 more' = 1 step\n"
                    "- 'Natalia sold 48 in April, half in May. Total?' = 2 steps\n"
                    "- 'Compound interest 7% quarterly 3 years $5000' = 4 steps\n"
                    "- 'Train 60mph 2.5hrs then 80mph 1.5hrs average speed' = 4 steps\n"
                )
            },
            {"role": "user", "content": request}
        ],
        temperature=0.0,
        max_tokens=5,
    )
    text = response.choices[0].message.content.strip()
    try:
        steps = int(text)
        # Chuyển số bước → score (ít bước = dễ = score cao)
        score = 1.0 if steps <= 2 else 0.0
        return score
    except:
        return 0.5

# Test
tests = [
    ("Tom has 5 apples. He buys 3 more. How many?",                                    1),
    ("Natalia sold 48 clips in April, half in May. Total?",                             1),
    ("Compound interest at 7% quarterly for 3 years on $5000. Final amount?",           0),
    ("A train travels 60mph for 2.5hrs then 80mph for 1.5hrs. Average speed?",          0),
    ("Lisa bought 3 books at $5 each. How much did she spend?",                         1),
    ("A store has 10% discount over 5 items, 8.5% tax. 8 apples at $1.2. Total cost?",  0),
]

print("Testing difficulty scorer...\n")
correct = 0
for req, expected in tests:
    score = get_difficulty_score(req)
    route = "SLM" if score >= 0.5 else "LLM"
    exp   = "SLM" if expected == 1 else "LLM"
    ok    = "OK" if route == exp else "WRONG"
    if route == exp:
        correct += 1
    print(f"{ok} score={score:.1f} expect={exp} got={route} | {req[:50]}")

print(f"\nAccuracy: {correct}/{len(tests)} = {correct/len(tests)*100:.0f}%")
