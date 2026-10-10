import ollama
import pandas as pd

MODEL = "qwen2.5-coder:3b"
FENCE = chr(96) * 3

# Load and prepare the data
df = pd.read_csv("sales-analysis.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True)
df["ship_date"] = pd.to_datetime(df["ship_date"], dayfirst=True)
df["year"] = df["order_date"].dt.year
df["month_name"] = df["order_date"].dt.month_name()


def make_prompt(question):
    return f"""You are a Python pandas expert.
A DataFrame called df already exists. Do not load or create data.

Columns and types:
{df.dtypes.to_string()}

First 3 rows:
{df.head(3).to_string()}

Write Python code that answers this question: {question}

Rules:
- Use only pandas and the existing df.
- Store the final answer in a variable named result. Include the numeric values, not just the labels.
- Return ONLY code. No explanation, no comments, no print."""


def clean_code(text):
    if FENCE in text:
        text = text.split(FENCE)[1]
        if text.startswith("python"):
            text = text[len("python"):]
    return text.strip()


def ask(question):
    prompt = make_prompt(question)
    code = ""
    last_error = ""
    for attempt in range(2):
        resp = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        code = clean_code(resp["message"]["content"])
        env = {"df": df.copy(), "pd": pd}
        try:
            exec(code, env)
            return code, env.get("result"), True
        except Exception as e:
            last_error = str(e)
            prompt += f"\n\nYour previous code failed with this error: {last_error}\nPrevious code:\n{code}\nFix it. Return ONLY code."
    return code, f"Could not answer. Last error: {last_error}", False


def explain(question, result):
    text = str(result)[:1500]
    prompt = f"""A user asked: {question}

The data analysis result is:
{text}

Explain the answer in 1 or 2 simple English sentences for a business person.
Use the numbers exactly as given. Do not add any facts that are not in the result."""
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip()


print("Ask questions about your sales data. Type exit to stop.")
while True:
    q = input("\nYour question: ").strip()
    if q.lower() == "exit":
        break
    if not q:
        continue
    code, result, ok = ask(q)
    print("\n--- Code written by AI ---")
    print(code)
    print("\n--- Answer ---")
    print(result)
    if ok:
        print("\n--- Explanation ---")
        print(explain(q, result))
