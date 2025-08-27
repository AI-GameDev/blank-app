"""
Streamlit + LangChain summarizer using GPT-4o-mini

Quick start (in a fresh environment):
    pip install --upgrade streamlit langchain langchain-openai tiktoken

Run:
    streamlit run streamlit_langchain_gpt4o_mini_summarizer.py

Notes:
- Your OpenAI API key is entered in the sidebar. It is NOT persisted.
- Model default is `gpt-4o-mini`, adjustable in the sidebar.
- Short texts use a single-pass prompt; long texts use map-reduce summarization.
"""

import os
import textwrap
from typing import List

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

APP_TITLE = "📝 LangChain Summarizer (GPT-4o-mini)"
DEFAULT_MODEL = "gpt-4o-mini"

# -----------------------------
# Utility: Build single-pass chain
# -----------------------------
def build_single_pass_chain(llm: ChatOpenAI, tone: str, length: str, language: str):
    """Return a function that takes raw text -> summary (single prompt)."""
    length_directive = {
        "짧게": "Keep it concise (2-3 sentences).",
        "보통": "Aim for a medium-length summary (4-7 sentences).",
        "길게": "Provide a detailed summary (8-12 sentences).",
    }[length]

    system = f"""
You are a helpful expert summarizer. Summarize the user's text in {language}.
Use a {tone} tone. {length_directive}
- Preserve key facts, entities, and numerical details.
- Avoid redundancy and personal opinions.
- Output only the summary, no preamble.
""".strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input_text}")
    ])

    def run(text: str) -> str:
        chain = prompt | llm
        resp = chain.invoke({"input_text": text})
        return resp.content

    return run


# -----------------------------
# Utility: Build map-reduce chain for long inputs
# -----------------------------
def build_map_reduce_chain(llm: ChatOpenAI, tone: str, length: str, language: str):
    """LangChain's load_summarize_chain(map_reduce) on Documents."""
    length_directive = {
        "짧게": "Keep it concise (bullet points or 2-3 sentences).",
        "보통": "Aim for a medium-length summary (4-7 sentences).",
        "길게": "Provide a detailed summary (8-12 sentences).",
    }[length]

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", f"Summarize the following chunk in {language} with a {tone} tone. {length_directive} Output only the summary."),
        ("human", "{text}")
    ])

    combine_prompt = ChatPromptTemplate.from_messages([
        ("system", f"Combine the chunk summaries into a single cohesive summary in {language}. Maintain a {tone} tone and {length_directive} Avoid repetition."),
        ("human", "{text}")
    ])

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=False,
        verbose=False,
    )

    def run(docs: List[Document]) -> str:
        out = chain.invoke({"input_documents": docs})
        return out["output_text"]

    return run


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("🔐 OpenAI 설정")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", help="키는 세션 동안만 사용되며 저장되지 않습니다.")

    st.header("⚙️ 모델 & 파라미터")
    model_name = st.text_input("Model", value=DEFAULT_MODEL, help="예: gpt-4o-mini, gpt-4o, o3-mini 등")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    st.header("🧭 요약 옵션")
    tone = st.selectbox("톤", ["중립적", "공식적", "친근한", "설명형", "분석적"])  # Korean tone options
    length = st.radio("길이", ["짧게", "보통", "길게"], index=1, horizontal=True)
    language = st.selectbox("요약 언어", ["한국어", "English", "日本語", "中文"])

    st.markdown("""
    **Tip**
    - 1만자 이상의 긴 텍스트는 자동으로 분할/병합(Map-Reduce) 방식으로 요약합니다.
    - 업로드된 `.txt` 파일도 지원합니다.
    """)

# Input area
col1, col2 = st.columns([2, 1])
with col1:
    input_text = st.text_area(
        "요약할 텍스트 입력",
        height=280,
        placeholder="여기에 긴 텍스트를 붙여넣거나 업로드를 사용하세요...",
    )
with col2:
    uploaded = st.file_uploader("또는 .txt 파일 업로드", type=["txt"])  # simple file support
    if uploaded is not None:
        try:
            file_text = uploaded.read().decode("utf-8", errors="ignore")
            # If user typed too, append a delimiter
            if input_text.strip():
                input_text += "\n\n" + file_text
            else:
                input_text = file_text
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")

# Validate API key
if not api_key:
    st.info("좌측 사이드바에 OpenAI API Key를 입력하세요.")
    st.stop()

# Prepare LLM
llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

# Run summarization
summarize_btn = st.button("🚀 요약 실행", type="primary")

if summarize_btn:
    if not input_text or not input_text.strip():
        st.warning("요약할 텍스트를 입력하거나 파일을 업로드해주세요.")
        st.stop()

    # Heuristic: choose map-reduce for long content
    CHAR_THRESHOLD = 8000

    if len(input_text) <= CHAR_THRESHOLD:
        runner = build_single_pass_chain(llm, tone=tone, length=length, language=language)
        with st.spinner("단일 패스 요약 중..."):
            try:
                summary = runner(input_text)
                st.success("완료")
                st.subheader("요약 결과")
                st.write(summary)
            except Exception as e:
                st.error(f"요약 중 오류가 발생했습니다: {e}")
    else:
        # Split and use map-reduce
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(input_text)
        docs = [Document(page_content=c) for c in chunks]

        runner = build_map_reduce_chain(llm, tone=tone, length=length, language=language)
        with st.spinner(f"긴 문서 분할({len(docs)} 청크) 후 요약 중..."):
            try:
                summary = runner(docs)
                st.success("완료")
                st.subheader("요약 결과")
                st.write(summary)
            except Exception as e:
                st.error(f"요약 중 오류가 발생했습니다: {e}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit · LangChain · OpenAI (gpt-4o-mini)")
