# requests_data = [

#     "What is recursion?",

#     "Explain why the sky is blue.",

#     "Write binary search in Python.",

#     "What causes inflation?",

#     "Solve 25 * 17.",

#     "Explain transformers in AI.",

#     "What is overfitting in machine learning?",

#     "Write a Python function for quicksort.",

#     "Why do airplanes fly?",

#     "Explain gravity simply."
# ]




# dataset.py
requests_data = [
    {"request": "What is 25 * 17?",                          "label": 1},
    {"request": "What is the capital of France?",             "label": 1},
    {"request": "Write a loop from 1 to 10 in Python.",       "label": 1},
    {"request": "What is recursion?",                         "label": 1},
    {"request": "Explain why the sky is blue.",               "label": 1},
    {"request": "What does len() do in Python?",              "label": 1},
    {"request": "How many seconds in an hour?",               "label": 1},
    {"request": "What year did World War II end?",            "label": 1},
    {"request": "Write a function to reverse a string.",      "label": 1},
    {"request": "What is the square root of 144?",            "label": 1},

    {"request": "Who directed Titanic and who was his mentor?",      "label": 0},
    {"request": "Implement a BST with insert, delete, search.",      "label": 0},
    {"request": "Compare microservices vs monolithic architecture.",  "label": 0},
    {"request": "Solve compound interest: $5000, 7%, 3 years.",      "label": 0},
    {"request": "Explain transformer architecture in deep learning.", "label": 0},
    {"request": "What are trade-offs of CAP theorem in databases?",   "label": 0},
    {"request": "Implement quicksort and explain time complexity.",   "label": 0},
    {"request": "Compare Keynesianism and Monetarism economics.",     "label": 0},
    {"request": "Write a thread-safe singleton pattern in Python.",   "label": 0},
    {"request": "Explain gradient descent with backpropagation.",     "label": 0},
]