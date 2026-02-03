import os
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# =========================
# 1️⃣ Initialize Flask
# =========================
app = Flask(__name__)

# =========================
# 2️⃣ Initialize Groq LLM
# =========================
llm = ChatGroq(
    api_key="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",
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# =========================
# 3️⃣ Prompt Template
# =========================
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a safe and helpful assistant. Answer concisely in a single line.

Rules:
- Do NOT provide medical, legal, or financial advice.
- Refuse unsafe questions politely.
- Respond clearly, factually, and precisely.

Question:
{question}

Answer:
"""
)

# =========================
# 4️⃣ Guardrails
# =========================
FORBIDDEN_KEYWORDS = ["kill", "suicide", "bomb", "hack", "weapon", "drug", "illegal"]
MEDICAL_KEYWORDS = ["heart attack", "symptoms", "disease", "treatment", "medicine", "diagnosis"]

def is_unsafe_question(question: str) -> bool:
    return any(word in question.lower() for word in FORBIDDEN_KEYWORDS)

def is_medical_question(question: str) -> bool:
    return any(word in question.lower() for word in MEDICAL_KEYWORDS)

# =========================
# 5️⃣ Token Handling / Question Summary
# =========================
def summarize_question(question: str) -> str:
    # For simplicity, truncate or summarize long questions
    max_words = 50
    words = question.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + "..."
        return summary
    return question

# =========================
# 6️⃣ Main Answer Function
# =========================
def answer_question(question: str) -> str:
    # Input validation
    if not question or not isinstance(question, str):
        return "Please provide a valid question."

    # Guardrails
    if is_unsafe_question(question):
        return "Sorry, I can't help with that request."
    if is_medical_question(question):
        return "I can’t provide medical advice. Please consult a qualified healthcare professional."

    # Token handling: summarize if too long
    question_summary = summarize_question(question)

    # Build prompt
    final_prompt = prompt.format(question=question_summary)

    try:
        # Call LLM
        response = llm.invoke(final_prompt)
        answer_text = getattr(response, "content", str(response))

        # Output guardrail: ensure concise answer
        if len(answer_text.strip()) < 10:
            return "Sorry, I am not able to answer this safely."

        # Logging
        print(f"[LOG] Question: {question}")
        print(f"[LOG] Answer: {answer_text}")

        return answer_text

    except Exception as e:
        print(f"[ERROR] {e}")
        return "Sorry, something went wrong while processing your question."

# =========================
# 7️⃣ Flask API Route
# =========================
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    answer = answer_question(question)
    return jsonify({"question": question, "answer": answer})

# =========================
# 8️⃣ Run Flask App
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
