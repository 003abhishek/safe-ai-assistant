import os
from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# =========================
# Load API Key securely
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# =========================
# Prompt
# =========================
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a safe AI assistant.
Answer in ONE clear factual line.

Rules:
- No medical, legal, financial advice
- Refuse unsafe questions politely

Question:
{question}

Answer:
"""
)

# =========================
# Guardrails
# =========================
FORBIDDEN = ["kill", "suicide", "bomb", "hack", "weapon", "drug", "illegal"]
MEDICAL = ["heart attack", "symptoms", "disease", "treatment", "medicine"]

def is_unsafe(q): return any(w in q.lower() for w in FORBIDDEN)
def is_medical(q): return any(w in q.lower() for w in MEDICAL)

def summarize_question(q, max_words=50):
    words = q.split()
    return " ".join(words[:max_words]) if len(words) > max_words else q

# =========================
# Core logic
# =========================
def get_answer(question):
    if not question or not isinstance(question, str):
        return "Please provide a valid question."

    if is_unsafe(question):
        return "Sorry, I can't help with that request."

    if is_medical(question):
        return "I can’t provide medical advice. Please consult a healthcare professional."

    question = summarize_question(question)
    final_prompt = prompt.format(question=question)

    try:
        response = llm.invoke(final_prompt)
        answer = getattr(response, "content", str(response))
        return answer.strip() if len(answer.strip()) > 10 else "Unable to answer safely."
    except Exception as e:
        print("[ERROR]", e)
        return "Internal error. Please try again later."

# =========================
# Routes
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    return jsonify({"answer": get_answer(question)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
