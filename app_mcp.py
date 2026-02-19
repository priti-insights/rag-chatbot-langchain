import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ---------------- RAG IMPORTS ----------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.tools import Tool


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RAG MCP Chat", layout="wide")
st.title("🤖 RAG Chat Application")


# ---------------- CHAT HISTORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- LOAD RAG ----------------
@st.cache_resource(show_spinner=True)
def load_rag():

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS index
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 2})

    # Load Qwen model
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

    # MCP-aware Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

If the user wants to export chat history as PDF,
respond EXACTLY with:
TOOL: generate_pdf

Otherwise answer the QUESTION strictly using the CONTEXT provided.

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


# ---------------- PDF GENERATOR ----------------
def generate_pdf(chat_history):
    file_path = "chat_history.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph("<b>RAG Chat History (MCP Export)</b>", styles["Title"])
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


# ---------------- MCP TOOL WRAPPER ----------------
def pdf_tool_function(_):
    pdf_file = generate_pdf(st.session_state.chat_history)
    return f"PDF generated successfully at {pdf_file}"


pdf_tool = Tool(
    name="generate_pdf",
    func=pdf_tool_function,
    description="Use this tool to export the entire chat history as a PDF file."
)


# ---------------- SIDEBAR ----------------
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


# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Ask a question about the speech...")

if user_input:
    with st.spinner("Thinking... 🤔"):

        response = rag_chain.invoke(user_input)

        # ---- MCP TOOL DETECTION ----
        if "TOOL: generate_pdf" in response:

            tool_result = pdf_tool.invoke("")

            with open("chat_history.pdf", "rb") as f:
                st.download_button(
                    label="⬇ Download PDF",
                    data=f,
                    file_name="chat_history.pdf",
                    mime="application/pdf"
                )

            final_answer = "📄 PDF generated successfully. Click above to download."

        else:
            final_answer = response

    st.session_state.chat_history.append(
        {"question": user_input, "answer": final_answer}
    )


# ---------------- DISPLAY CHAT ----------------
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])

    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
