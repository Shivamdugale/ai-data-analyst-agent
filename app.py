import ollama
import pandas as pd
import streamlit as st

MODEL = "qwen2.5-coder:3b"
FENCE = chr(96) * 3

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊")
st.title("📊 AI Data Analyst Agent")
st.caption("Ask questions about your CSV in plain English. Runs fully on your laptop with a free local model.")


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    date_cols = [c for c in df.columns if "date" in c]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
    if date_cols:
        df["year"] = df[date_cols[0]].dt.year
        df["month_name"] = df[date_cols[0]].dt.month_name()
    return df


def make_prompt(df, question):
    return f"""You are a Python pandas expert.
A DataFrame called df already exists. Do not load or create data.

Columns and types:
{df.dtypes.to_string()}

First 3 rows:
{df.head(3).to_string()}

Write Python code that answers this question: {question}

Rules:
- Use only pandas and the existing df.
- Store the final answer in a variable named result.
- Include the numeric values, not just the labels.
- Return ONLY code. No explanation, no comments, no print."""


def clean_code(text):
    if FENCE in text:
        text = text.split(FENCE)[1]
        if text.startswith("python"):
            text = text[len("python"):]
    return text.strip()


def ask(df, question):
    prompt = make_prompt(df, question)
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


def get_chart_data(result):
    if isinstance(result, pd.Series) and len(result) > 1 and pd.api.types.is_numeric_dtype(result):
        return result
    if isinstance(result, pd.DataFrame) and result.shape[1] == 2 and pd.api.types.is_numeric_dtype(result.iloc[:, 1]):
        return result.set_index(result.columns[0]).iloc[:, 0]
    return None


uploaded = st.file_uploader("Upload a CSV file (or leave empty to use the sample sales data)", type="csv")
df = load_data(uploaded if uploaded else "sales-analysis.csv")
st.write(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
with st.expander("Preview data"):
    st.dataframe(df.head())

question = st.text_input("Your question", placeholder="Which region has the highest total sales?")

if st.button("Ask") and question:
    with st.spinner("AI is writing and running the analysis..."):
        code, result, ok = ask(df, question)

    st.subheader("Answer")
    if isinstance(result, (pd.Series, pd.DataFrame)):
        st.dataframe(result)
    else:
        st.write(result)

    chart_data = get_chart_data(result) if ok else None
    if chart_data is not None:
        st.bar_chart(chart_data)

    if ok:
        with st.spinner("Writing a simple explanation..."):
            st.subheader("Explanation")
            st.write(explain(question, result))

    with st.expander("Code written by AI"):
        st.code(code, language="python")
