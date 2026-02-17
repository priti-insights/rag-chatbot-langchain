import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

# ---- RAG IMPORTS ----
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ---------------------------
# Load Vectorstore
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 2})

    model_name = "Qwen/Qwen2-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.0,
        do_sample=False,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the QUESTION strictly using the CONTEXT provided.
If the answer is not present in the context, say:
"I cannot find the answer in the given context."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = load_rag()

# ---------------------------
# Export to PDF
# ---------------------------
def generate_pdf(chat_history):
    file_path = "chat_history.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph("<b>RAG Chat History</b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 24))

    for i, chat in enumerate(chat_history, 1):
        q = Paragraph(f"<b>Q{i}:</b> {chat['question']}", styles["Normal"])
        a = Paragraph(f"<b>A{i}:</b> {chat['answer']}", styles["Normal"])

        elements.append(q)
        elements.append(Spacer(1, 12))
        elements.append(a)
        elements.append(Spacer(1, 24))

    doc.build(elements)
    return file_path

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="RAG Chat", layout="wide")

st.title("🤖 RAG Chat Application")

# Sidebar
with st.sidebar:
    st.header("⚙️ Options")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")

    st.markdown("---")

    if st.button("📄 Export Chat as PDF"):
        if st.session_state.chat_history:
            pdf_file = generate_pdf(st.session_state.chat_history)
            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="⬇ Download PDF",
                    data=f,
                    file_name="chat_history.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("No chat history to export.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input (ChatGPT style)
user_input = st.chat_input("Ask a question about the speech...")

if user_input:
    with st.spinner("Thinking... 🤔"):
        answer = rag_chain.invoke(user_input)

    st.session_state.chat_history.append(
        {"question": user_input, "answer": answer}
    )

# Display chat messages
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])

    with st.chat_message("assistant"):
        st.markdown(chat["answer"])


