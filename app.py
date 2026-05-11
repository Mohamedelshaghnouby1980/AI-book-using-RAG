import streamlit as st
import sys
import os

sys.path.insert(0, "src")

st.set_page_config(
    page_title="AI Books RAG",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.source-card {
    background: #f8f9fa;
    border-left: 3px solid #0066cc;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 4px;
    font-size: 0.85em;
}
.score-badge {
    background: #0066cc;
    color: white;
    padding: 2px 6px;
    border-radius: 10px;
    font-size: 0.75em;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


@st.cache_resource
def load_rag():
    from main import RAGSystem
    return RAGSystem()


with st.sidebar:
    st.title("📚 AI Books RAG")
    st.caption("using  Ollama model")
    st.divider()

    st.subheader("إضافة كتاب")
    uploaded_pdf = st.file_uploader("ارفع ملف PDF", type=["pdf"])

    if uploaded_pdf and st.button("إضافة الكتاب", type="primary"):
        save_path = f"data/books/{uploaded_pdf.name}"
        os.makedirs("data/books", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_pdf.read())
        rag = load_rag()
        with st.spinner("جاري معالجة الكتاب..."):
            rag.ingest_book(save_path)
        st.success("الكتاب جاهز!")

    st.divider()

    if st.button("مسح المحادثة"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()


col1, col2 = st.columns([2, 1])

with col1:
    st.title("📚 AI Books RAG")
    st.caption("اسأل أي سؤال عن كتبك")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("اكتب سؤالك هنا..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("جاري البحث..."):
                try:
                    rag = load_rag()
                    result = rag.ask(query)
                    answer = result["answer"]
                    st.session_state.last_sources = result["sources"]
                except Exception as e:
                    answer = f"خطأ: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

with col2:
    st.subheader("المصادر")
    if st.session_state.last_sources:
        for src in st.session_state.last_sources:
            st.markdown(f"""
<div class="source-card">
    <span class="score-badge">{src['score']:.2f}</span>
    <b>صفحة {src['page']}</b><br>
    <small>{src['source']}</small>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("المصادر هتظهر هنا بعد ما تسأل سؤال.")